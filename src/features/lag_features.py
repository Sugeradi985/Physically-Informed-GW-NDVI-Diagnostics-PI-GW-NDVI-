# src/features/lag_features.py

import xarray as xr
from typing import Dict, List, Optional

def add_lagged_vars(
    ds: xr.Dataset,
    lag_config: Dict[str, List[int]],
    target_time: Optional[xr.DataArray] = None
) -> xr.Dataset:
    """
    为 xarray.Dataset 中的指定变量添加滞后特征。

    Parameters
    ----------
    ds : xr.Dataset
        输入数据集，包含原始变量（如 'NDVI', 'Precip'）
    lag_config : dict
        滞后配置，例如 {'NDVI': [1, 2], 'Precip': [0, 1, 2]}
        - key: 变量名
        - value: 滞后阶数列表（正整数表示过去，0 表示当前）
    target_time : xr.DataArray or None
        可选：目标时间轴（如 ΔGW 的时间），用于截断对齐

    Returns
    -------
    xr.Dataset
        包含原始变量 + 所有滞后变量的新数据集
    """
    ds_out = ds.copy()

    for var_name, lags in lag_config.items():
        if var_name not in ds:
            raise KeyError(f"Variable '{var_name}' not found in dataset.")
        
        for lag in lags:
            if lag < 0:
                raise ValueError("Lag steps must be >= 0.")
            # lag=0 表示当前时刻（可选，用于显式包含）
            shifted = ds[var_name].shift(time=lag) if lag > 0 else ds[var_name]
            new_name = f"{var_name}_lag{lag}" if lag > 0 else var_name
            ds_out[new_name] = shifted

    # 如果提供了 target_time，对齐并截断到共同时间范围
    if target_time is not None:
        common_time = ds_out.time.where(ds_out.time.isin(target_time.time), drop=True)
        ds_out = ds_out.sel(time=common_time)

    return ds_out
