import numpy as np
import pandas as pd
import math
from .utils import validate_input

# Constants
SOLAR_CONSTANT = 0.0820  # MJ m-2 min-1
STEFAN_BOLTZMANN = 4.903e-9  # MJ K-4 m-2 day-1
ALBEDO = 0.23  # Standard albedo for reference grass crop

def _get_day_stats(dates):
    """Helper to extract day of year and days in month."""
    doy = dates.dayofyear.values
    days_in_month = dates.days_in_month.values
    return doy, days_in_month

def _calc_solar_declination(doy):
    """Calculates solar declination (radians). FAO eq 24."""
    return 0.409 * np.sin((2.0 * np.pi / 365.0) * doy - 1.39)

def _calc_sunset_hour_angle(lat_rad, declination):
    """Calculates sunset hour angle (radians). FAO eq 25."""
    val = -np.tan(lat_rad) * np.tan(declination)
    val = np.clip(val, -1.0, 1.0)
    return np.arccos(val)

def _calc_extraterrestrial_radiation(doy, lat_rad, declination, ws):
    """Calculates Ra (MJ m-2 day-1). FAO eq 21."""
    dr = 1 + 0.033 * np.cos((2.0 * np.pi / 365.0) * doy)
    ra = (24.0 * 60.0 / np.pi) * SOLAR_CONSTANT * dr * (
        ws * np.sin(lat_rad) * np.sin(declination) +
        np.cos(lat_rad) * np.cos(declination) * np.sin(ws)
    )
    return np.maximum(ra, 0)

def thornthwaite(tmean: np.ndarray, dates: pd.DatetimeIndex, lat: float) -> np.ndarray:
    """
    Calculates Monthly Potential Evapotranspiration (PET) using the Thornthwaite (1948) method.
    
    Parameters:
    -----------
    tmean : np.ndarray
        Mean monthly temperature (Celsius).
    dates : pd.DatetimeIndex
        Dates corresponding to the data (Must be Monthly frequency).
    lat : float
        Latitude in degrees.
        
    Returns:
    --------
    pet : np.ndarray
        Monthly PET (mm/month).
    """
    t = validate_input(tmean)
    lat_rad = np.radians(lat)
    
    # 1. Annual Heat Index (I)
    # We need to sum (T/5)^1.514 for each year.
    # Create a DataFrame to group by year easily
    df = pd.DataFrame({'t': t, 'year': dates.year})
    
    # Clip negative temperatures to 0 for heat index calculation
    t_idx = np.maximum(df['t'], 0)
    df['i_contrib'] = (t_idx / 5.0) ** 1.514
    
    annual_I = df.groupby('year')['i_contrib'].transform('sum').values
    
    # 2. Exponent (a)
    a = (6.75e-7 * annual_I**3) - (7.71e-5 * annual_I**2) + (1.792e-2 * annual_I) + 0.49239
    
    # 3. Daylight Hours Correction (L)
    doy = dates.dayofyear.values + 15 # Approximate middle of month
    declination = _calc_solar_declination(doy)
    ws = _calc_sunset_hour_angle(lat_rad, declination)
    daylight_hours = (24.0 / np.pi) * ws
    
    days_in_month = dates.days_in_month.values
    
    # 4. Calculate PET
    # PET_unadjusted = 16 * (10 * T / I)^a
    # Adjustment = (L / 12) * (N / 30)
    
    pet_unadj = 16 * np.power((10 * t_idx.values / annual_I), a)
    pet = pet_unadj * (daylight_hours / 12.0) * (days_in_month / 30.0)
    
    return pet

def hargreaves(tmin: np.ndarray, tmax: np.ndarray, tmean: np.ndarray, 
               dates: pd.DatetimeIndex, lat: float) -> np.ndarray:
    """
    Calculates Daily PET using the Hargreaves (1985) method.
    Good when solar radiation, wind, or humidity data are missing.
    
    Parameters:
    -----------
    tmin, tmax, tmean : np.ndarray
        Daily temperatures (Celsius).
    dates : pd.DatetimeIndex
        Daily dates.
    lat : float
        Latitude in degrees.
        
    Returns:
    --------
    pet : np.ndarray
        Daily PET (mm/day).
    """
    tmin = validate_input(tmin)
    tmax = validate_input(tmax)
    tmean = validate_input(tmean)
    lat_rad = np.radians(lat)
    doy = dates.dayofyear.values
    
    # 1. Extraterrestrial Radiation (Ra)
    declination = _calc_solar_declination(doy)
    ws = _calc_sunset_hour_angle(lat_rad, declination)
    ra = _calc_extraterrestrial_radiation(doy, lat_rad, declination, ws)
    
    # 2. Hargreaves Equation
    # 0.408 converts MJ/m2/day to mm/day
    # PET = 0.0023 * (Tmean + 17.8) * (Tmax - Tmin)^0.5 * Ra
    
    trange = tmax - tmin
    trange[trange < 0] = 0 # Physics check
    
    pet = 0.0023 * (tmean + 17.8) * np.sqrt(trange) * 0.408 * ra
    
    return pet

def priestley_taylor(tmean: np.ndarray, net_radiation: np.ndarray, 
                    dates: pd.DatetimeIndex, elevation: float = 0, alpha: float = 1.26) -> np.ndarray:
    """
    Calculates PET using the Priestley-Taylor (1972) method.
    Suitable for humid conditions where aerodynamic term is less important.
    
    Parameters:
    -----------
    tmean : np.ndarray
        Mean daily temperature (Celsius).
    net_radiation : np.ndarray
        Net radiation (Rn) in MJ m-2 day-1.
    dates : pd.DatetimeIndex
        Daily dates.
    elevation : float
        Elevation in meters (default 0).
    alpha : float
        PT coefficient (default 1.26).
        
    Returns:
    --------
    pet : np.ndarray
        Daily PET (mm/day).
    """
    t = validate_input(tmean)
    rn = validate_input(net_radiation)
    
    # Psychrometric Constant (gamma) [kPa C-1]
    # P = 101.3 * ((293 - 0.0065*z) / 293)^5.26
    pressure = 101.3 * np.power((293 - 0.0065 * elevation) / 293, 5.26)
    gamma = 0.000665 * pressure
    
    # Slope of Vapor Pressure Curve (delta) [kPa C-1]
    # delta = 4098 * (0.6108 * exp(17.27 * T / (T + 237.3))) / (T + 237.3)^2
    tmp = 4098 * (0.6108 * np.exp((17.27 * t) / (t + 237.3)))
    delta = tmp / np.power((t + 237.3), 2)
    
    # Ground Heat Flux (G)
    # Usually 0 for daily steps
    G = 0 
    
    # Equation
    # PET = alpha * (delta / (delta + gamma)) * (Rn - G) / lambda
    # lambda (latent heat of vaporization) is approx 2.45 MJ/kg
    
    pet = alpha * (delta / (delta + gamma)) * (rn - G) / 2.45
    return pet

def penman_monteith(tmin: np.ndarray, tmax: np.ndarray, tmean: np.ndarray,
                   rh_mean: np.ndarray, wind_speed: np.ndarray, net_radiation: np.ndarray,
                   elevation: float, lat: float) -> np.ndarray:
    """
    Calculates PET using the FAO-56 Penman-Monteith equation.
    The most physically robust method.
    
    Parameters:
    -----------
    tmin, tmax, tmean : np.ndarray
        Temperatures (Celsius).
    rh_mean : np.ndarray
        Relative Humidity (Percent, 0-100).
    wind_speed : np.ndarray
        Wind speed at 2m height (m/s).
    net_radiation : np.ndarray
        Net Radiation (Rn) in MJ m-2 day-1.
    elevation : float
        Elevation (meters).
    lat : float
        Latitude (degrees).
        
    Returns:
    --------
    pet : np.ndarray
        Daily PET (mm/day).
    """
    # 1. Slope of Saturation Vapor Pressure Curve (delta)
    tmp = 4098 * (0.6108 * np.exp((17.27 * tmean) / (tmean + 237.3)))
    delta = tmp / np.power((tmean + 237.3), 2)
    
    # 2. Psychrometric Constant (gamma)
    pressure = 101.3 * np.power((293 - 0.0065 * elevation) / 293, 5.26)
    gamma = 0.000665 * pressure
    
    # 3. Vapor Pressure Deficit (es - ea)
    # es = (e(Tmax) + e(Tmin)) / 2
    e_tmax = 0.6108 * np.exp((17.27 * tmax) / (tmax + 237.3))
    e_tmin = 0.6108 * np.exp((17.27 * tmin) / (tmin + 237.3))
    es = (e_tmax + e_tmin) / 2.0
    
    # Actual vapor pressure
    ea = (rh_mean / 100.0) * es
    
    # 4. Wind Speed (u2)
    # Assumed input is already at 2m. If not, user must convert.
    u2 = validate_input(wind_speed)
    
    # 5. The Equation (FAO-56 Eq 6)
    # NUM = 0.408*delta*(Rn-G) + gamma*(900/(T+273))*u2*(es-ea)
    # DEN = delta + gamma*(1 + 0.34*u2)
    
    rn = validate_input(net_radiation)
    G = 0 # Daily soil heat flux assumed 0
    
    num1 = 0.408 * delta * (rn - G)
    num2 = gamma * (900 / (tmean + 273)) * u2 * (es - ea)
    den = delta + gamma * (1 + 0.34 * u2)
    
    pet = (num1 + num2) / den
    
    return pet