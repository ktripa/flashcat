import numpy as np
import pandas as pd
from scipy.stats import rankdata
from ..utils import validate_input, daily_to_pentad

def calc_smvi_metrics(rzsm: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Calculates metrics required for SMVI (Osman et al. 2021).
    
    Metrics:
    1. Pentad-mean RZSM.
    2. 4-Pentad (20-day) Running Average.
    3. Climatological Percentiles for each pentad.
    
    Parameters:
    -----------
    rzsm : np.ndarray
        Daily Root Zone Soil Moisture.
    dates : pd.DatetimeIndex
        Dates corresponding to the data.
        
    Returns:
    --------
    df_pentad : pd.DataFrame
        Columns: ['year', 'pentad', 'rzsm_mean', 'rolling_avg', 'percentile']
    """
    rzsm_vals = validate_input(rzsm)
    
    # 1. Aggregate Daily -> Pentad
    df_pentad = daily_to_pentad(rzsm_vals, dates)
    df_pentad.rename(columns={'val': 'rzsm_mean'}, inplace=True)
    
    # 2. Calculate 4-Pentad Running Average (Equation 12)
    # "Short-term soil moisture conditions"
    # We use a trailing window of 4 pentads (current + 3 previous)
    # If the current pentad is part of the drying, it will pull the average down slightly,
    # but a rapid drop will still result in Current < Average.
    df_pentad['rolling_avg'] = df_pentad['rzsm_mean'].rolling(window=4, min_periods=4).mean()
    
    # 3. Calculate Climatological Percentiles
    df_pentad['percentile'] = np.nan
    
    for p in range(1, 74):
        p_mask = df_pentad['pentad'] == p
        vals = df_pentad.loc[p_mask, 'rzsm_mean']
        
        valid_idx = vals.dropna().index
        
        if len(valid_idx) > 5:
            # Rank data (Low RZSM = Low Rank)
            ranks = rankdata(vals[valid_idx], method='average')
            n = len(vals[valid_idx])
            
            # Percentile = (Rank / N) * 100
            pcts = (ranks / n) * 100
            df_pentad.loc[valid_idx, 'percentile'] = pcts
            
    return df_pentad

def identify_flash_drought(df_pentad: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies flash drought using SMVI Dual-Threshold criteria.
    
    Criteria (Osman et al.):
    1. Volatility: RZSM < 4-Pentad Rolling Average.
    2. Drought: RZSM < 20th Percentile.
    3. Duration: Both conditions met for at least 4 consecutive pentads.
    """
    # Thresholds
    THRESH_PCT = 20.0
    MIN_DURATION = 4 # Consecutive pentads
    
    # Ensure sorted order
    df = df_pentad.sort_values(['year', 'pentad']).reset_index(drop=True)
    
    # Extract arrays for speed
    rzsm = df['rzsm_mean'].values
    rolling = df['rolling_avg'].values
    pcts = df['percentile'].values
    years = df['year'].values
    pentads = df['pentad'].values
    n = len(df)
    
    events = []
    
    # Scan for consecutive periods where BOTH criteria are met
    # Condition: (RZSM < Rolling) AND (Percentile < 20)
    
    current_seq_start = None
    seq_length = 0
    
    for i in range(n):
        val = rzsm[i]
        avg = rolling[i]
        pct = pcts[i]
        
        if np.isnan(val) or np.isnan(avg) or np.isnan(pct):
            # Break sequence on missing data
            if seq_length >= MIN_DURATION:
                # Save event before resetting
                start_idx = current_seq_start
                end_idx = i - 1
                events.append({
                    'start_year': years[start_idx],
                    'start_pentad': pentads[start_idx],
                    'end_year': years[end_idx],
                    'end_pentad': pentads[end_idx],
                    'duration_pentads': seq_length,
                    'min_percentile': np.min(pcts[start_idx:end_idx+1])
                })
            current_seq_start = None
            seq_length = 0
            continue
            
        # CHECK DUAL CONDITIONS
        condition_met = (val < avg) and (pct < THRESH_PCT)
        
        if condition_met:
            if current_seq_start is None:
                current_seq_start = i
                seq_length = 1
            else:
                seq_length += 1
        else:
            # Condition broken. Check if we had a valid event.
            if seq_length >= MIN_DURATION:
                start_idx = current_seq_start
                end_idx = i - 1
                
                events.append({
                    'start_year': years[start_idx],
                    'start_pentad': pentads[start_idx],
                    'end_year': years[end_idx],
                    'end_pentad': pentads[end_idx],
                    'duration_pentads': seq_length,
                    'min_percentile': np.min(pcts[start_idx:end_idx+1])
                })
            
            # Reset
            current_seq_start = None
            seq_length = 0
            
    # Handle case where event goes until the very end of the time series
    if seq_length >= MIN_DURATION:
        start_idx = current_seq_start
        end_idx = n - 1
        events.append({
            'start_year': years[start_idx],
            'start_pentad': pentads[start_idx],
            'end_year': years[end_idx],
            'end_pentad': pentads[end_idx],
            'duration_pentads': seq_length,
            'min_percentile': np.min(pcts[start_idx:end_idx+1])
        })

    return pd.DataFrame(events)