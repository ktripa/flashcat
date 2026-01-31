"""
Module for computing the Self-Calibrated Palmer Drought Severity Index (scPDSI).
References: Wells et al. (2004), Palmer (1965).
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
from ..utils import validate_input

__all__ = ["calc_scpdsi"]

AWCTOP_MM = 25.4 

def _get_awc_bot(awc_mm: float) -> float:
    return max(awc_mm - AWCTOP_MM, 0.0)

def _calc_potential_loss(pet_mm: float, ss_mm: float, su_mm: float, awc_mm: float) -> float:
    awc_bot = _get_awc_bot(awc_mm)
    if ss_mm >= pet_mm:
        return pet_mm
    return min(ss_mm + su_mm, ((pet_mm - ss_mm) * su_mm) / (awc_bot + AWCTOP_MM) + ss_mm)

def _calc_water_balance(precip_mm: np.ndarray, pet_mm: np.ndarray, awc_mm: float) -> dict:
    n = len(precip_mm)
    awc_bot = _get_awc_bot(awc_mm)
    ss = AWCTOP_MM
    su = awc_bot
    
    components = {
        "et": np.zeros(n), "r": np.zeros(n), "ro": np.zeros(n), 
        "l": np.zeros(n), "pe": pet_mm, "p": precip_mm
    }
    
    potentials = {
        "pr": np.zeros(n), "pl": np.zeros(n), "pro": np.zeros(n)
    }

    for i in range(n):
        p = precip_mm[i]
        pe = pet_mm[i]
        
        potentials["pr"][i] = (AWCTOP_MM - ss) + (awc_bot - su)
        potentials["pl"][i] = _calc_potential_loss(pe, ss, su, awc_mm)
        potentials["pro"][i] = ss + su 

        if p >= pe:
            et = pe
            loss = 0.0
            excess = p - pe
            
            if excess > (AWCTOP_MM - ss):
                recharge_s = AWCTOP_MM - ss
                ss_new = AWCTOP_MM
                excess -= recharge_s
                
                if excess < (awc_bot - su):
                    recharge_u = excess
                    su_new = su + recharge_u
                    ro = 0.0
                else:
                    recharge_u = awc_bot - su
                    su_new = awc_bot
                    ro = excess - recharge_u
                
                r = recharge_s + recharge_u
            else:
                r = excess
                ss_new = ss + excess
                su_new = su
                ro = 0.0
        else:
            r = 0.0
            ro = 0.0
            deficit = pe - p
            
            if ss >= deficit:
                loss_s = deficit
                loss_u = 0.0
                ss_new = ss - loss_s
                su_new = su
            else:
                loss_s = ss
                ss_new = 0.0
                needed = deficit - loss_s
                loss_u = min(su, (needed * su) / awc_mm)
                su_new = su - loss_u
            
            loss = loss_s + loss_u
            et = p + loss
            
        components["et"][i] = et
        components["r"][i] = r
        components["ro"][i] = ro
        components["l"][i] = loss
        
        ss = ss_new
        su = su_new
        
    return components, potentials

def _calc_cafec(components: dict, potentials: dict, dates: pd.DatetimeIndex) -> dict:
    coeffs = {"alpha": np.zeros(12), "beta": np.zeros(12), 
              "gamma": np.zeros(12), "delta": np.zeros(12)}
    
    df_comp = pd.DataFrame(components)
    df_comp["month"] = dates.month - 1
    df_pot = pd.DataFrame(potentials)
    df_pot["month"] = dates.month - 1
    
    sums_comp = df_comp.groupby("month").sum()
    sums_pot = df_pot.groupby("month").sum()
    
    for m in range(12):
        pe_sum = sums_comp.loc[m, "pe"]
        coeffs["alpha"][m] = sums_comp.loc[m, "et"] / pe_sum if pe_sum > 0 else 1.0
        
        pr_sum = sums_pot.loc[m, "pr"]
        coeffs["beta"][m] = sums_comp.loc[m, "r"] / pr_sum if pr_sum > 0 else 1.0
        
        pro_sum = sums_pot.loc[m, "pro"]
        coeffs["gamma"][m] = sums_comp.loc[m, "ro"] / pro_sum if pro_sum > 0 else 1.0
        
        pl_sum = sums_pot.loc[m, "pl"]
        coeffs["delta"][m] = sums_comp.loc[m, "l"] / pl_sum if pl_sum > 0 else 1.0
        
    return coeffs

def _calibrate_parameters(z_index: np.ndarray) -> tuple:
    durations = [3, 6, 9, 12, 18, 24]
    dry_ratio = []
    wet_ratio = []
    
    for k in durations:
        z_accum = pd.Series(z_index).rolling(window=k).sum().dropna().values
        if len(z_accum) < 20: 
            dry_ratio.append(-4.0)
            wet_ratio.append(4.0)
            continue
            
        dry_extremes = np.percentile(z_accum, 2)
        wet_extremes = np.percentile(z_accum, 98)
        dry_ratio.append(dry_extremes)
        wet_ratio.append(wet_extremes)
        
    res_dry = linregress(durations, dry_ratio)
    res_wet = linregress(durations, wet_ratio)
    
    m_dry, b_dry = res_dry.slope, res_dry.intercept
    m_wet, b_wet = res_wet.slope, res_wet.intercept
    
    denom_dry = m_dry + b_dry
    p_dry = 1 - (m_dry / denom_dry) if denom_dry != 0 else 0.897
    q_dry = -4.0 / denom_dry if denom_dry != 0 else 1.0/3.0
    
    denom_wet = m_wet + b_wet
    p_wet = 1 - (m_wet / denom_wet) if denom_wet != 0 else 0.897
    q_wet = 4.0 / denom_wet if denom_wet != 0 else 1.0/3.0
    
    p = (p_dry + p_wet) / 2.0
    q = (abs(q_dry) + abs(q_wet)) / 2.0
    
    return p, q

def calc_scpdsi(precip_mm: np.ndarray, pet_mm: np.ndarray, awc_mm: float, dates: pd.DatetimeIndex) -> tuple:
    """
    Computes the Self-Calibrated Palmer Drought Severity Index (scPDSI).
    
    Parameters:
    -----------
    precip_mm : np.ndarray
        Monthly precipitation time series (mm).
    pet_mm : np.ndarray
        Monthly potential evapotranspiration time series (mm).
    awc_mm : float
        Available Water Capacity of the soil (mm).
    dates : pd.DatetimeIndex
        Dates corresponding to the data (Monthly frequency).
        
    Returns:
    --------
    tuple (pdsi, phdi, z_index)
        pdsi : np.ndarray - The Self-Calibrated PDSI.
        phdi : np.ndarray - The Hydrological Drought Index.
        z_index : np.ndarray - The moisture anomaly Z-index.
    """
    precip = validate_input(precip_mm)
    pet = validate_input(pet_mm)
    
    comps, pots = _calc_water_balance(precip, pet, awc_mm)
    
    coeffs = _calc_cafec(comps, pots, dates)
    
    n = len(precip)
    d = np.zeros(n)
    k_constants = np.zeros(12)
    
    months = dates.month - 1
    
    df_d_temp = pd.DataFrame({"pe": pots["pe"], "p": comps["p"], "l": comps["l"], "month": months})
    
    for i in range(n):
        m = months[i]
        p_hat = (coeffs["alpha"][m] * pots["pe"][i] + 
                 coeffs["beta"][m] * pots["pr"][i] + 
                 coeffs["gamma"][m] * pots["pro"][i] - 
                 coeffs["delta"][m] * pots["pl"][i])
        d[i] = comps["p"][i] - p_hat
        
    df_d_temp["d"] = d
    means = df_d_temp.groupby("month").mean()
    
    for m in range(12):
        pe = means.loc[m, "pe"]
        p = means.loc[m, "p"]
        l = means.loc[m, "l"]
        d_abs = abs(means.loc[m, "d"])
        
        if d_abs == 0: d_abs = 1.0
        
        k_prime = 1.5 * np.log10(( (pe + p + l) / d_abs ) + 2.8) + 0.5
        k_constants[m] = k_prime
        
    sum_dk = np.sum(means["d"].abs() * k_constants)
    if sum_dk == 0: sum_dk = 1.0
    
    k_vals = (17.67 / sum_dk) * k_constants
    z_index = d * k_vals[months]
    
    p_factor, q_factor = _calibrate_parameters(z_index)
    
    x = np.zeros(n)
    phdi = np.zeros(n)
    x_prev = 0.0
    
    for i in range(n):
        x_curr = p_factor * x_prev + q_factor * z_index[i]
        x_curr = max(min(x_curr, 10.0), -10.0)
        x[i] = x_curr
        x_prev = x_curr
        phdi[i] = x[i]
        
    return x, phdi, z_index