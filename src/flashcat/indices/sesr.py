import numpy as np
import pandas as pd
from scipy import signal
from ..utils import validate_input, daily_to_pentad

def calc_sesr(et: np.ndarray, pet: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Calculates the Standardized Evaporative Stress Ratio (SESR) and Delta-SESR.
    
    Methodology (Christian et al., 2019):
    1. Calculate ESR = ET / PET daily.
    2. Aggregate to Pentads (5-day means).
    3. Detrend ESR.
    4. Standardize ESR to get SESR (Z-score).
    5. Calculate Change (Delta) in SESR.
    6. Standardize Delta to get Delta-SESR (Z-score).
    
    Returns:
    --------
    result : pd.DataFrame
        Columns: ['year', 'pentad', 'sesr', 'delta_sesr']
    """
    et_vals = validate_input(et)
    pet_vals = validate_input(pet)
    
    # 1. Calculate Daily ESR
    with np.errstate(divide='ignore', invalid='ignore'):
        esr_daily = et_vals / pet_vals
    
    # Clean Data
    esr_daily[np.isinf(esr_daily)] = np.nan
    esr_daily[esr_daily < 0] = 0
    esr_daily[esr_daily > 2] = 2.0  # Physical cap
    
    # 2. Aggregate to Pentads
    # We aggregate ET and PET first? Or ESR? 
    # Manuscript Equation 8 implies aggregating the ratio or ratio of aggregations.
    # Christian et al 2019 usually aggregates the daily ESR values.
    # Legacy code aggregates the daily ESR. We follow that.
    df_esr = daily_to_pentad(esr_daily, dates)
    
    # 3. Detrending (Linear) - Per Grid
    # Since this function handles one grid/time-series, we detrend the whole series.
    valid_mask = ~np.isnan(df_esr['val'])
    if valid_mask.sum() > 10:
        df_esr.loc[valid_mask, 'val'] = signal.detrend(df_esr.loc[valid_mask, 'val'], type='linear')
        
    # 4. Standardize ESR -> SESR (Equation 9)
    # Standardize by Pentad (compare Pentad 1 only to other Pentad 1s)
    sesr_values = np.full(len(df_esr), np.nan)
    
    for p in range(1, 74):
        p_mask = df_esr['pentad'] == p
        vals = df_esr.loc[p_mask, 'val']
        
        if len(vals) > 5 and np.nanstd(vals) > 1e-6:
            mean_p = np.nanmean(vals)
            std_p = np.nanstd(vals, ddof=1)
            sesr_values[p_mask] = (vals - mean_p) / std_p
            
    df_esr['sesr'] = sesr_values
    
    # 5. Calculate Delta SESR (Equation 10)
    # Change from previous pentad
    df_esr['delta_raw'] = df_esr['sesr'].diff()
    
    # Handle year boundaries (Pentad 1 - Prev Year Pentad 73)
    # Pandas diff() handles this naturally if data is sorted chronologically
    
    # 6. Standardize Delta (Equation 11)
    delta_z = np.full(len(df_esr), np.nan)
    
    for p in range(1, 74):
        p_mask = df_esr['pentad'] == p
        vals = df_esr.loc[p_mask, 'delta_raw']
        
        if len(vals) > 5 and np.nanstd(vals) > 1e-6:
            mean_d = np.nanmean(vals)
            std_d = np.nanstd(vals, ddof=1)
            delta_z[p_mask] = (vals - mean_d) / std_d
            
    df_esr['delta_sesr'] = delta_z
    
    return df_esr[['year', 'pentad', 'sesr', 'delta_sesr']]

def identify_flash_drought(df_sesr: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies flash drought using SESR (Christian et al. 2019).
    
    Criteria:
    1. Duration: Min 5 consecutive changes (6 pentads).
    2. Final Intensity: SESR < 20th percentile.
    3. Rate of Change: Delta-SESR < 40th percentile (Max 1 violation allowed).
    4. Mean Rate: Mean Delta-SESR < 25th percentile.
    """
    # Calculate Percentile Thresholds specific to this time series
    
    vals_sesr = df_sesr['sesr'].dropna()
    vals_delta = df_sesr['delta_sesr'].dropna()
    
    if len(vals_sesr) < 30:
        return pd.DataFrame()

    thresh_final = np.percentile(vals_sesr, 20)      # Final SESR < 20th
    thresh_delta_indiv = np.percentile(vals_delta, 40) # Each Delta < 40th
    thresh_delta_mean = np.percentile(vals_delta, 25)  # Mean Delta < 25th
    
    events = []

    
    potential_start = None
    current_seq = [] # Stores indices of the sequence
    violations = 0
    MAX_VIOLATIONS = 1
    MIN_CHANGES = 5 #  6 pentads duration
    
    # Arrays for fast access
    sesr_arr = df_sesr['sesr'].values
    delta_arr = df_sesr['delta_sesr'].values
    years = df_sesr['year'].values
    pentads = df_sesr['pentad'].values
    n = len(df_sesr)
    
    for i in range(1, n):
        d_val = delta_arr[i]
        s_val = sesr_arr[i]
        
        if np.isnan(d_val) or np.isnan(s_val):
            potential_start = None
            current_seq = []
            violations = 0
            continue
            
        # Check if this step qualifies as part of a rapid decline
        # Condition: Delta is negative (decline) AND below threshold
        is_rapid = (d_val < 0) and (d_val <= thresh_delta_indiv)
        
        if is_rapid:
            if potential_start is None:
                potential_start = i
                current_seq = [i]
                violations = 0
            else:
                current_seq.append(i)
                
        elif potential_start is not None:
            # Not rapid, but check if we can burn a violation
            if violations < MAX_VIOLATIONS:
                violations += 1
                current_seq.append(i)
            else:
                # Sequence ends. Validate Event.
                # Valid if:
                # 1. Length >= 5 changes
                # 2. Final SESR < 20th percentile
                # 3. Mean Delta < 25th percentile
                
                if len(current_seq) >= MIN_CHANGES:
                    # The sequence indices represent the "changes" (deltas)
                    # Event duration covers pentads: [Start-1] to [End]
                    # because Delta[i] is change from i-1 to i.
                    
                    final_idx = current_seq[-1]
                    final_sesr_val = sesr_arr[final_idx]
                    
                    deltas_in_event = delta_arr[current_seq]
                    mean_event_delta = np.mean(deltas_in_event)
                    
                    if (final_sesr_val <= thresh_final) and (mean_event_delta <= thresh_delta_mean):
                        # RECORD EVENT
                        start_idx = current_seq[0] - 1 # The pentad before the first drop
                        end_idx = final_idx
                        
                        events.append({
                            'start_year': years[start_idx],
                            'start_pentad': pentads[start_idx],
                            'end_year': years[end_idx],
                            'end_pentad': pentads[end_idx],
                            'duration_pentads': len(current_seq) + 1,
                            'final_sesr': final_sesr_val,
                            'mean_delta_sesr': mean_event_delta
                        })
                
                # Reset
                potential_start = None
                current_seq = []
                violations = 0

    return pd.DataFrame(events)