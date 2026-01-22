import numpy as np
import pandas as pd
from scipy.stats import rankdata
from ..utils import validate_input, daily_to_pentad

def calc_smpd_percentiles(rzsm: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Calculates Pentad-mean RZSM Percentiles for SMPD.
    
    Parameters:
    -----------
    rzsm : np.ndarray
        Daily Root Zone Soil Moisture.
    dates : pd.DatetimeIndex
        Dates corresponding to the data.
        
    Returns:
    --------
    df_pentad : pd.DataFrame
        Columns: ['year', 'pentad', 'rzsm_mean', 'percentile']
    """
    rzsm_vals = validate_input(rzsm)
    
    # 1. Aggregate Daily -> Pentad (5-day means)
    # Using the shared utility from utils.py
    df_pentad = daily_to_pentad(rzsm_vals, dates)
    df_pentad.rename(columns={'val': 'rzsm_mean'}, inplace=True)
    
    # 2. Calculate Percentiles (Pentad-based Climatology)
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
    Identifies flash drought using SMPD criteria (Ford & Labosier, 2017).
    
    Criteria:
    1. Onset: Drop from >= 40th to <= 20th percentile within 4 pentads (Rapid Descent).
    2. Sustain: RZSM remains <= 20th percentile.
    3. Recovery: RZSM > 20th percentile for 3 CONSECUTIVE pentads (15 days).
       (If it rises above 20 but drops back within 3 pentads, detection continues).
    """
    # Thresholds
    THRESH_START = 40.0
    THRESH_DROUGHT = 20.0
    MAX_ONSET_WINDOW = 4 # Pentads (must drop 40->20 within this time)
    RECOVERY_REQ = 3     # Pentads (must stay recovered for 3 steps to end)
    
    # Sort and extract arrays
    df = df_pentad.sort_values(['year', 'pentad']).reset_index(drop=True)
    pcts = df['percentile'].values
    years = df['year'].values
    pentads = df['pentad'].values
    n = len(df)
    
    events = []
    
    in_drought = False
    onset_start_idx = None # Index where it was last >= 40
    drought_start_idx = None # Index where it hit <= 20
    recovery_counter = 0   # Track consecutive recovered pentads
    
    for i in range(1, n):
        curr_p = pcts[i]
        
        if np.isnan(curr_p):
            in_drought = False
            recovery_counter = 0
            continue
            
        if not in_drought:
            # CHECK ONSET TRIGGER (Hit <= 20)
            if curr_p <= THRESH_DROUGHT:
                # We hit drought. Did we come from >= 40 quickly?
                # Scan backwards up to MAX_ONSET_WINDOW
                found_start = False
                valid_start_idx = -1
                
                # We need to find the *most recent* time it was >= 40
                # within the last 4 pentads
                for lookback in range(1, MAX_ONSET_WINDOW + 1):
                    idx = i - lookback
                    if idx < 0: break
                    
                    if pcts[idx] >= THRESH_START:
                        found_start = True
                        valid_start_idx = idx
                        break # Found the start of the drop
                
                if found_start:
                    # VALID FLASH DROUGHT START
                    in_drought = True
                    drought_start_idx = i
                    onset_start_idx = valid_start_idx
                    recovery_counter = 0
                    
        else: # IN DROUGHT
            # CHECK RECOVERY (Must exceed 20 for 3 consecutive steps)
            if curr_p > THRESH_DROUGHT:
                recovery_counter += 1
                
                if recovery_counter >= RECOVERY_REQ:
                    # CONFIRMED END OF DROUGHT
                    # The drought effectively ended 3 pentads ago (when it first crossed)
                    # or we count the recovery period? 
                    # Usually, end date is the first pentad of recovery.
                    end_idx = i - RECOVERY_REQ + 1
                    
                    # Calculate Metrics
                    duration = end_idx - onset_start_idx
                    
                    # Calculate Onset Speed (Percentiles per pentad)
                    onset_time = drought_start_idx - onset_start_idx
                    drop_mag = pcts[onset_start_idx] - pcts[drought_start_idx]
                    speed = drop_mag / max(1, onset_time)
                    
                    events.append({
                        'start_year': years[onset_start_idx],
                        'start_pentad': pentads[onset_start_idx],
                        'end_year': years[end_idx],
                        'end_pentad': pentads[end_idx],
                        'duration_pentads': duration,
                        'onset_speed': speed, # Percentiles/pentad
                        'min_percentile': np.min(pcts[drought_start_idx:end_idx+1])
                    })
                    
                    in_drought = False
                    recovery_counter = 0
            else:
                # DIPPED BACK DOWN into drought
                # Reset recovery counter (it wasn't sustained)
                recovery_counter = 0

    return pd.DataFrame(events)