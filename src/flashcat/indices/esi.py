import numpy as np
import pandas as pd
from scipy.stats import norm
from ..utils import validate_input, get_empirical_probability

def calc_esi(et: np.ndarray, pet: np.ndarray, dates: pd.DatetimeIndex, window: int = 14) -> pd.Series:
    """
    Calculates the Evaporative Stress Index (ESI).
    
    ESI = Standardized Anomaly of (ET / PET) ratio.
    Interpretation: Negative values indicate drought (ET < PET).
    
    Parameters:
    -----------
    et : np.ndarray
        Actual Evapotranspiration daily time series.
    pet : np.ndarray
        Potential Evapotranspiration daily time series.
    dates : pd.DatetimeIndex
        Dates corresponding to the data.
    window : int
        Aggregation window in days (default 14).
        
    Returns:
    --------
    esi_z : pd.Series
        Time series of ESI Z-scores.
    """
    et_vals = validate_input(et)
    pet_vals = validate_input(pet)
    
    # Check dimensions
    if len(et_vals) != len(pet_vals):
        raise ValueError("ET and PET arrays must have the same length.")
    if window not in [14, 28]:
        raise ValueError("The current package supports ESI-14 and ESI-28 only.")

    # Step 1: Temporal Aggregation (Running Means)
    # Using trailing window (standard for monitoring)
    et_series = pd.Series(et_vals, index=dates)
    pet_series = pd.Series(pet_vals, index=dates)
    
    et_agg = et_series.rolling(window=window, min_periods=window).mean()
    pet_agg = pet_series.rolling(window=window, min_periods=window).mean()
    
    # Step 2: Calculate Evaporative Stress Ratio (ESR)
    # ESR = ET / PET
    # Handle division by zero: If PET is 0, ESR is undefined (NaN)
    with np.errstate(divide='ignore', invalid='ignore'):
        esr = et_agg.values / pet_agg.values
    
    # Clean Infinite values (if PET=0) and negative values (physically impossible)
    esr[np.isinf(esr)] = np.nan
    esr[esr < 0] = 0
    esr[esr > 2] = 2 # Clip unrealistic ratios (ET can slightly exceed PET in oasis effects, but rarely > 2)
    
    # Step 3: Standardization (Inverse Normal)
    # Low ESR (Drought) -> Low Rank -> Negative Z-score
    esi_z = get_empirical_probability(esr, dates)
    
    return pd.Series(esi_z, index=dates, name=f'ESI_{window}')

def identify_flash_drought(esi_z: pd.Series, window: int = 14) -> pd.DataFrame:
    """
    Identifies flash drought events using ESI (Anderson et al., Nguyen et al.).
    
    Direction: DROUGHT IS LOW VALUES (Negative Z-scores).
    
    Criteria:
    1. Onset: Rapid DECLINE of >= 50 percentile points in 'window' days.
    2. Threshold: Falls below 20th percentile.
    3. Duration: Stays below 20th percentile for at least 'window' days.
    4. Termination: Rises above 20th percentile and stays there.
    """
    dates = esi_z.index
    
    # 1. Convert Z-scores to Percentiles (0-100)
    pct = norm.cdf(esi_z.values) * 100
    pct_series = pd.Series(pct, index=dates)

    # 2. Calculate Change (Delta)
    # delta = Current - Past
    # For ESI, we want a large NEGATIVE delta (Drop)
    delta = pct_series.diff(periods=window)

    events = []
    in_event = False
    onset_idx = 0
    
    # Thresholds (from your legacy code & manuscript)
    THRESH_DROUGHT = 20.0   # Drought state (<= 20th)
    DROP_CRITERIA = 50.0    # Rapid drop magnitude
    MIN_DURATION = window   # 14 days
    RECOVERY_WINDOW = 14    # Must stay recovered for 14 days to count as end

    for i in range(window, len(pct)):
        current_pct = pct[i]
        current_change = delta.iloc[i] # Current - Past
        
        if np.isnan(current_pct) or np.isnan(current_change):
            continue

        if not in_event:
            # CHECK ONSET
            # 1. Rapid Drop: (Past - Current) >= 50  <==> Current - Past <= -50
            # 2. Drought State: Current <= 20
            
            # Note: current_change is (Current - Past). A drop of 50 means current_change <= -50.
            if (current_change <= -DROP_CRITERIA) and (current_pct <= THRESH_DROUGHT):
                in_event = True
                onset_idx = i
                
        else:
            # CHECK TERMINATION
            # Ends if it rises ABOVE 20th percentile
            if current_pct > THRESH_DROUGHT:
                # Check for SUSTAINED recovery (look ahead 14 days)
                # If it dips back down within 14 days, it's not over.
                is_recovered = True
                look_ahead = min(i + RECOVERY_WINDOW, len(pct))
                
                if np.any(pct[i:look_ahead] <= THRESH_DROUGHT):
                    is_recovered = False
                
                if is_recovered:
                    # Event officially ends at 'i'
                    event_end_idx = i
                    duration_days = (dates[event_end_idx] - dates[onset_idx]).days
                    
                    # Verify Duration Requirement
                    if duration_days >= MIN_DURATION:
                        events.append({
                            'onset_date': dates[onset_idx],
                            'end_date': dates[event_end_idx],
                            'duration_days': duration_days,
                            'min_intensity_pct': np.nanmin(pct[onset_idx:event_end_idx]),
                            'mean_intensity_pct': np.nanmean(pct[onset_idx:event_end_idx])
                        })
                    
                    in_event = False

    if not events:
        return pd.DataFrame(columns=['onset_date', 'end_date', 'duration_days', 'min_intensity_pct'])

    return pd.DataFrame(events)