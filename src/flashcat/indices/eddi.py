import numpy as np
import pandas as pd
from scipy.stats import norm
from ..utils import validate_input, get_empirical_probability

def calc_eddi(pet: np.ndarray, dates: pd.DatetimeIndex, window: int = 14) -> pd.Series:
    """
    Calculates EDDI (Evaporative Demand Drought Index).
    Returns Z-scores (Standard Deviations).
    """
    if window not in [14, 28]:
        raise ValueError("EDDI usually supports 14 or 28 day windows.")

    pet_vals = validate_input(pet)
    
    # Step 1: Temporal Aggregation (14-day Running Mean)
    # Uses 'min_periods' to allow calculation even if a few days are missing
    pet_series = pd.Series(pet_vals, index=dates)
    pet_agg = pet_series.rolling(window=window, min_periods=window).mean()

    # Step 2: Standardization (Inverse Normal)
    # This calls our corrected util function
    eddi_z = get_empirical_probability(pet_agg.values, dates)
    
    return pd.Series(eddi_z, index=dates, name=f'EDDI_{window}')

def identify_flash_drought(eddi_z_series: pd.Series, window: int = 14) -> pd.DataFrame:
    """
    Identifies flash drought events based on Pendergrass et al. (2020) / Parker et al. (2021).
    """
    dates = eddi_z_series.index
    
    # 1. Convert Z-scores to Percentiles (0-100) for thresholding
    # This matches your logic: >= 80th percentile
    pct = norm.cdf(eddi_z_series.values) * 100
    pct_series = pd.Series(pct, index=dates)

    # 2. Calculate the change over the window (Delta)
    delta = pct_series.diff(periods=window)

    events = []
    in_event = False
    onset_idx = 0
    
    # Thresholds
    THRESH_80 = 80.0   # Severity threshold
    DELTA_50 = 50.0    # Rapid intensification threshold
    MIN_DURATION = window # Usually 14 or 28 days

    # Iterate through time (vectorized where possible, but loop needed for state)
    for i in range(window, len(pct)):
        current_pct = pct[i]
        current_delta = delta.iloc[i]
        
        if np.isnan(current_pct) or np.isnan(current_delta):
            continue

        if not in_event:
            # CHECK ONSET:
            # Rapid rise (>=50 percentile points) AND High Intensity (>=80th)
            if (current_delta >= DELTA_50) and (current_pct >= THRESH_80):
                in_event = True
                onset_idx = i
                
        else:
            # CHECK TERMINATION:
            # Validates if the event has ended (dropped below 80th)
            # Simplification: If it drops below 80, we check if it STAYS below.
            if current_pct < THRESH_80:
                # We assume the event ends the day it drops below 80
                event_end_idx = i - 1
                duration_days = (dates[event_end_idx] - dates[onset_idx]).days
                
                # Filter: Must meet minimum duration (e.g., 14 days)
                if duration_days >= MIN_DURATION:
                    events.append({
                        'onset_date': dates[onset_idx],
                        'end_date': dates[event_end_idx],
                        'duration_days': duration_days,
                        'max_intensity_pct': np.nanmax(pct[onset_idx:event_end_idx+1]),
                        'mean_intensity_pct': np.nanmean(pct[onset_idx:event_end_idx+1])
                    })
                
                in_event = False

    if not events:
        return pd.DataFrame(columns=['onset_date', 'end_date', 'duration_days', 'max_intensity_pct'])

    return pd.DataFrame(events)