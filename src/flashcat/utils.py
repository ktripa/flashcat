import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from typing import Union

def validate_input(data: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
    if isinstance(data, list):
        return np.array(data)
    elif isinstance(data, (pd.Series, pd.DataFrame)):
        return data.values
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(f"Input type {type(data)} not supported. Use numpy or pandas.")
    
def get_empirical_probability(data: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Calculates standardized Z-scores using the Inverse Normal Approximation (Hobbins et al. 2016).
    """
    df = pd.DataFrame({'val': data, 'doy': dates.dayofyear})
    result = np.full(data.shape, np.nan)
    groups = df.groupby('doy')

    for _, group_idx in groups.indices.items():
        
        values = data[group_idx]
        
        valid_mask = ~np.isnan(values)
        if np.sum(valid_mask) < 3: # Need at least 3 years of data
            continue

        # 1. Rank the data (Tukey plotting position)
        ranks = rankdata(values[valid_mask], method='average')
        n = len(values[valid_mask])
        probs = (ranks - 0.33) / (n + 0.33)
        z_scores = norm.ppf(probs)
        result[group_idx[valid_mask]] = z_scores

    return result

def daily_to_pentad(data: np.ndarray, dates: pd.DatetimeIndex) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregates daily data to Pentads (73 periods per year).
    Pentad 73 includes remaining days (Days 361-365/366).
    
    Returns:
    --------
    pentad_data : pd.DataFrame
        Aggregated values (Year, Pentad, Value).
    """
    df = pd.DataFrame({'val': data, 'year': dates.year})
    
    # Create Pentad index (1-73)
    # Day of year 1-5 -> Pentad 1, ..., 361+ -> Pentad 73
    doy = dates.dayofyear
    pentad_idx = np.ceil(doy / 5).astype(int)
    pentad_idx = np.clip(pentad_idx, 1, 73) # Ensure days >365 go to 73
    
    df['pentad'] = pentad_idx
    
    # Group by Year and Pentad
    # Taking the mean (ignoring NaNs)
    aggregated = df.groupby(['year', 'pentad'])['val'].mean().reset_index()
    
    return aggregated
