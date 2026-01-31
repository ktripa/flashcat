import numpy as np
import pandas as pd
from scipy.stats import gamma, norm
from ..utils import validate_input

def calc_spi(precip: np.ndarray, dates: pd.DatetimeIndex, scale: int = 3, 
             calibration_start_year: int = None, calibration_end_year: int = None) -> pd.Series:
    """
    Calculates the Standardized Precipitation Index (SPI).
    
    Method:
    1. Aggregates precipitation over 'scale' months.
    2. Fits a Gamma distribution to the data for each calendar month.
    3. Handles zero-precipitation months using a mixed distribution probability.
    4. Transforms cumulative probability to Z-score.
    
    Parameters:
    -----------
    precip : np.ndarray
        Monthly precipitation time series (mm).
    dates : pd.DatetimeIndex
        Dates corresponding to the data (Monthly frequency).
    scale : int
        Timescale in months (e.g., 1, 3, 6, 12).
    calibration_start_year : int (Optional)
        Start year to calculate the distribution parameters.
    calibration_end_year : int (Optional)
        End year to calculate the distribution parameters.
        
    Returns:
    --------
    spi : pd.Series
        Time series of SPI values.
    """
    vals = validate_input(precip)
    
    # 1. Rolling Sum (Aggregation)
    # Use trailing window (standard for SPI)
    series = pd.Series(vals, index=dates)
    
    # Skip calculation if scale is larger than data length
    if len(series) < scale:
        raise ValueError("Data length is shorter than the requested SPI scale.")

    # Rolling Sum
    aggregated = series.rolling(window=scale, min_periods=scale).sum()
    
    # 2. Fit Distribution per Month
    # We calculate parameters based on the calibration period (or full period if None)
    if calibration_start_year and calibration_end_year:
        mask_calib = (dates.year >= calibration_start_year) & (dates.year <= calibration_end_year)
    else:
        mask_calib = np.ones(len(dates), dtype=bool)
        
    spi_final = pd.Series(np.nan, index=dates)
    
    # Group by month (1=Jan, 12=Dec)
    for month in range(1, 13):
        # Indices for this month
        month_mask = (dates.month == month)
        
        # Get data for this month
        month_data_all = aggregated[month_mask]
        
        # Get calibration data (filtering out NaNs created by rolling window)
        calib_data = aggregated[month_mask & mask_calib].dropna()
        
        if len(calib_data) < 10:
            # Not enough data to fit distribution
            continue
            
        # --- Handle Zeros (Mixed Distribution) ---
        # H(x) = q + (1-q) * G(x)
        # q = probability of zero
        n_zeros = (calib_data == 0).sum()
        n_total = len(calib_data)
        q = n_zeros / n_total
        
        # Fit Gamma to NON-ZERO data only
        data_nonzero = calib_data[calib_data > 0]
        
        if len(data_nonzero) > 0:
            # Fit Gamma (alpha=shape, loc, beta=scale)
            # We fix loc=0 because precip cannot be negative
            alpha, loc, beta = gamma.fit(data_nonzero, floc=0)
            
            # Calculate CDF for ALL data in this month (using parameters from calibration)
            # 1. CDF of Gamma part
            cdf_gamma = gamma.cdf(month_data_all, alpha, loc=0, scale=beta)
            
            # 2. Combined CDF
            # If value is 0, CDF is q. If >0, CDF is q + (1-q)*GammaCDF
            cdf_final = q + (1 - q) * cdf_gamma
            
            # Fix floating point issues (CDF must be < 1 for infinite Z)
            cdf_final[cdf_final >= 1.0] = 0.99999999
            cdf_final[cdf_final <= 0.0] = 0.00000001
            
            # Handle the exact zeros separately to be safe (they should be 'q')
            cdf_final[month_data_all == 0] = q if q > 0 else 0.00000001
            
        else:
            # If all days are zero (e.g., desert in dry season)
            # If it's always zero, we can't really define a deviation.
            # Usually return 0 or NaN.
            cdf_final = np.full(len(month_data_all), 0.5) # Median
        
        # 3. Transform to Z-score (Inverse Normal)
        z_scores = norm.ppf(cdf_final)
        
        # Store in result
        spi_final.loc[month_mask] = z_scores
        
    return spi_final.rename(f'SPI_{scale}')