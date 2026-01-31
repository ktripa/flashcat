import numpy as np
import pandas as pd
from scipy.stats import fisk, norm
from ..utils import validate_input
from .. import pet as pet_mod  # Import our PET module

def calc_spei(precip: np.ndarray, dates: pd.DatetimeIndex, scale: int = 3,
              pet: np.ndarray = None, 
              temp: np.ndarray = None, lat: float = None,
              calibration_start_year: int = None, 
              calibration_end_year: int = None) -> pd.Series:
    """
    Calculates the Standardized Precipitation Evapotranspiration Index (SPEI).
    
    Method:
    1. Calculates Water Balance D = Precipitation - PET.
    2. Aggregates D over 'scale' months.
    3. Fits a Log-Logistic (Fisk) distribution to D.
    4. Transforms to Z-scores.
    
    **Auto-PET Feature:**
    If 'pet' is None, but 'temp' and 'lat' are provided, this function 
    automatically calculates PET using the Thornthwaite method.
    
    Parameters:
    -----------
    precip : np.ndarray
        Monthly precipitation (mm).
    dates : pd.DatetimeIndex
        Dates (Monthly).
    scale : int
        Timescale in months.
    pet : np.ndarray (Optional)
        Monthly Potential Evapotranspiration (mm).
    temp : np.ndarray (Optional)
        Mean Monthly Temperature (Celsius). Required if PET is missing.
    lat : float (Optional)
        Latitude. Required if PET is missing (for Thornthwaite).
        
    Returns:
    --------
    spei : pd.Series
        Time series of SPEI values.
    """
    p_vals = validate_input(precip)
    
    # --- Step 1: Handle PET Input ---
    if pet is None:
        if temp is None or lat is None:
            raise ValueError("If 'pet' is not provided, you MUST provide 'temp' (mean temperature) and 'lat' (latitude) to calculate it internally.")
        
        # Auto-Calculate PET using Thornthwaite (Default for SPEI when data limited)
        print("FlashCAT Info: PET not provided. Calculating using Thornthwaite method...")
        pet_vals = pet_mod.thornthwaite(temp, dates, lat)
    else:
        pet_vals = validate_input(pet)
        
    # --- Step 2: Calculate Water Balance (D) ---
    # D = P - PET
    d_vals = p_vals - pet_vals
    
    # --- Step 3: Rolling Sum (Aggregation) ---
    series_d = pd.Series(d_vals, index=dates)
    
    if len(series_d) < scale:
        raise ValueError("Data length is shorter than the requested SPEI scale.")

    aggregated = series_d.rolling(window=scale, min_periods=scale).sum()
    
    # --- Step 4: Fit Distribution (Log-Logistic / Fisk) ---
    # SPEI uses Log-Logistic because D can be negative.
    
    if calibration_start_year and calibration_end_year:
        mask_calib = (dates.year >= calibration_start_year) & (dates.year <= calibration_end_year)
    else:
        mask_calib = np.ones(len(dates), dtype=bool)
        
    spei_final = pd.Series(np.nan, index=dates)
    
    for month in range(1, 13):
        month_mask = (dates.month == month)
        month_data_all = aggregated[month_mask]
        calib_data = aggregated[month_mask & mask_calib].dropna()
        
        if len(calib_data) < 10:
            continue
            
        # Fit Log-Logistic (Fisk)
        # Fisk in Scipy is 3-parameter: c (shape), loc (location), scale.
        # This handles the negative values in D by shifting the location.
        try:
            # We use Unbiased Probability Weighted Moments (PWM) or simple MLE?
            # SciPy uses MLE. For very short records, L-moments are better, 
            # but MLE is standard in Python packages.
            c, loc, scale_param = fisk.fit(calib_data)
            
            # Calculate CDF
            cdf = fisk.cdf(month_data_all, c, loc=loc, scale=scale_param)
            
            # Clip bounds to avoid Inf
            cdf[cdf >= 1.0] = 0.99999999
            cdf[cdf <= 0.0] = 0.00000001
            
            # Transform to Z-score
            z_scores = norm.ppf(cdf)
            spei_final.loc[month_mask] = z_scores
            
        except Exception:
            # Fallback if fit fails (rare with Fisk but possible on weird data)
            spei_final.loc[month_mask] = np.nan
            
    return spei_final.rename(f'SPEI_{scale}')