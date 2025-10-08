# src/features/auto_lag_selection.py
"""
Automatically select significant lag steps using autocorrelation function (ACF).
"""

import xarray as xr
import numpy as np
from statsmodels.tsa.stattools import acf

def suggest_lags_from_acf(
    da: xr.DataArray,
    max_lag: int = 6,
    threshold: float = 0.2,
    min_lag: int = 1
) -> list:
    """
    Suggest lag steps based on ACF of spatially averaged time series.

    Parameters
    ----------
    da : xr.DataArray
        Input variable with dimensions (time, lat, lon)
    max_lag : int
        Maximum lag to consider
    threshold : float
        ACF absolute value threshold for significance
    min_lag : int
        Minimum lag to return (default=1)

    Returns
    -------
    list of int
        Recommended lag steps (e.g., [1, 2, 4])
    """
    # Spatial average to get representative time series
    ts = da.mean(dim=['lat', 'lon'], skipna=True).values

    # Remove NaNs at ends (common in GW data)
    ts = ts[~np.isnan(ts)]
    if len(ts) < max_lag + 10:
        print(f"Warning: Time series too short ({len(ts)}). Using default lags.")
        return list(range(min_lag, min_lag + 2))

    try:
        acf_vals = acf(ts, nlags=max_lag, fft=True, missing='drop')
    except Exception as e:
        print(f"ACF failed: {e}. Using default lags.")
        return list(range(min_lag, min_lag + 2))

    lags = []
    for lag in range(min_lag, max_lag + 1):
        if abs(acf_vals[lag]) > threshold:
            lags.append(lag)

    if not lags:
        # Fallback: at least include first lag
        lags = [min_lag]

    return lags
