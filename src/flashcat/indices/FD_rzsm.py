import numpy as np
import pandas as pd
from scipy.stats import rankdata
from ..utils import validate_input, daily_to_pentad

def calc_rzsm_percentiles(rzsm: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Calculates Pentad-mean RZSM Percentiles for Flash Drought Detection.
    
    Steps:
    1. Aggregates Daily RZSM to Pentad Means.
    2. Calculates Percentiles based on historical climatology for each pentad.
    
    Parameters:
    -----------
    rzsm : np.ndarray
        Daily Root Zone Soil Moisture (volumetric or index).
    dates : pd.DatetimeIndex
        Dates corresponding to the data.
        
    Returns:
    --------
    df_pentad : pd.DataFrame
        Columns: ['year', 'pentad', 'rzsm_mean', 'percentile']
    """
    rzsm_vals = validate_input(rzsm)
    
    # 1. Aggregate Daily -> Pentad
    # Using the shared utility from utils.py
    df_pentad = daily_to_pentad(rzsm_vals, dates)
    df_pentad.rename(columns={'val': 'rzsm_mean'}, inplace=True)
    
    # 2. Calculate Percentiles (Pentad-based Climatology)
    # We compare Pentad 1 only with other Pentad 1s in history.
    df_pentad['percentile'] = np.nan
    
    for p in range(1, 74):
        p_mask = df_pentad['pentad'] == p
        vals = df_pentad.loc[p_mask, 'rzsm_mean']
        
        # Valid mask to ignore NaNs
        valid_idx = vals.dropna().index
        
        if len(valid_idx) > 5: # Need minimal history
            # Rank data (Low RZSM = Low Rank)
            # method='average' handles ties
            ranks = rankdata(vals[valid_idx], method='average')
            n = len(vals[valid_idx])
            
            # Standard Percentile Formula: (Rank / (N+1)) * 100
            # Or (Rank-0.33)/(N+0.33) if using Tukey. 
            # Yuan et al. typically use standard empirical percentiles.
            # We will use Rank/(N) * 100 to match standard hydrologic percentiles (0-100)
            pcts = (ranks / n) * 100
            
            df_pentad.loc[valid_idx, 'percentile'] = pcts
            
    return df_pentad

def identify_flash_drought(df_pentad: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies flash drought using Yuan et al. (2019) RZSM criteria.
    
    Criteria:
    1. Onset: Drop from >40th to <20th percentile.
    2. Rate: Average decline rate >= 5 percentile points per pentad during onset.
    3. Duration: Total event (Onset + Recovery) >= 4 pentads (20 days).
    4. Termination: Rises above 20th percentile.
    """
    # Thresholds
    THRESH_START = 40.0
    THRESH_DROUGHT = 20.0
    MIN_RATE = 5.0 # Percentiles per pentad
    MIN_DURATION = 4 # Pentads
    
    # We need sequential access
    df = df_pentad.sort_values(['year', 'pentad']).reset_index(drop=True)
    pcts = df['percentile'].values
    years = df['year'].values
    pentads = df['pentad'].values
    
    events = []
    
    # State Machine
    in_event = False
    onset_start_idx = None # Index where it dropped below 40
    drought_start_idx = None # Index where it hit <20 (Event fully triggered)
    
    # We iterate looking for the CROSSING of 40th downward
    
    for i in range(1, len(df)):
        curr_p = pcts[i]
        prev_p = pcts[i-1]
        
        if np.isnan(curr_p):
            in_event = False
            onset_start_idx = None
            continue
            
        # CHECK ONSET TRIGGER
        if not in_event:
            # We look for the transition from >40 to... eventually <20
            # Ideally, we track "last time it was >40"
            
            # Logic: If we are <= 20, we check backwards to see if we came from >40 rapidly
            if curr_p <= THRESH_DROUGHT:
                # Trace back to find when it crossed 40
                # Scan backwards max e.g. 10 pentads to find the start of the drop
                found_start = False
                scan_idx = i - 1
                
                while scan_idx >= 0 and (i - scan_idx) < 12: # Reasonable search window
                    if pcts[scan_idx] > THRESH_START:
                        found_start = True
                        onset_start_idx = scan_idx
                        break
                    scan_idx -= 1
                
                if found_start:
                    # Calculate Decline Rate
                    # Rate = (Start_Pct - Curr_Pct) / time_steps
                    delta_t = i - onset_start_idx
                    drop_amount = pcts[onset_start_idx] - curr_p
                    rate = drop_amount / delta_t
                    
                    if rate >= MIN_RATE:
                        # Valid Onset! Start Event.
                        in_event = True
                        # The event strictly starts when it drops below 40 (onset phase)
                        # But typically "Drought" counts from the <20 trigger?
                        # Yuan 2019 counts the whole "rapid drydown" as the onset phase.
                        # So start_idx is onset_start_idx.
                        
        # CHECK TERMINATION
        elif in_event:
            # Event ends if RZSM rises > 20
            if curr_p > THRESH_DROUGHT:
                # Event Over. Validate Duration.
                # Duration = End - Start (Onset Start)
                duration = i - onset_start_idx
                
                if duration >= MIN_DURATION:
                    # Calculate metrics
                    onset_phase_end = -1
                    min_val = 100
                    
                    # Find end of onset phase (lowest point or when rate slows?)
                    # Yuan 2019: Onset stage is "period when RZSM actively decreases 
                    # and rate >= 5".
                    # Simplified: Onset ends at the minimum percentile point.
                    
                    subset = pcts[onset_start_idx:i]
                    min_val = np.min(subset)
                    # Get index of min value (relative to subset)
                    min_rel_idx = np.argmin(subset)
                    onset_phase_end_idx = onset_start_idx + min_rel_idx
                    
                    events.append({
                        'start_year': years[onset_start_idx],
                        'start_pentad': pentads[onset_start_idx],
                        'end_year': years[i],
                        'end_pentad': pentads[i], # The pentad it recovered
                        'duration_pentads': duration,
                        'onset_pentads': (onset_phase_end_idx - onset_start_idx),
                        'recovery_pentads': (i - onset_phase_end_idx),
                        'decline_rate': (pcts[onset_start_idx] - min_val) / max(1, (onset_phase_end_idx - onset_start_idx)),
                        'min_percentile': min_val
                    })
                
                in_event = False
                onset_start_idx = None

    return pd.DataFrame(events)