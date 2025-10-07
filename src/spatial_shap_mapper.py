# spatial_shap_mapper.py
import xarray as xr
import numpy as np
import pandas as pd
import shap
from pathlib import Path

def generate_spatial_shap_maps(model, aligned_data, feature_cols, lags=[1, 2], output_dir="shap_maps/"):
    """
    为每个像元生成 SHAP 贡献值，并保存为 NetCDF
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # 构建完整时空样本（含滞后）
    gw = aligned_data["DeltaGW"]
    ndvi = aligned_data["NDVI"]
    precip = aligned_data.get("Precip")
    temp = aligned_data.get("Temp")

    # 展平构建特征矩阵（保留时空索引）
    df = pd.DataFrame({
        "time": gw.time.values.repeat(gw.y.size * gw.x.size),
        "y": np.tile(gw.y.values, gw.time.size * gw.x.size),
        "x": np.tile(np.repeat(gw.x.values, gw.y.size), gw.time.size),
        "NDVI": ndvi.values.flatten(),
        "DeltaGW": gw.values.flatten()
    })
    if precip is not None:
        df["Precip"] = precip.values.flatten()
    if temp is not None:
        df["Temp"] = temp.values.flatten()

    df = df.dropna().reset_index(drop=True)
    df = add_lagged_features(df, "NDVI", lags)
    df = df.dropna().reset_index(drop=True)

    X = df[feature_cols].values

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # shape: (n_samples, n_features)

    # 重建为时空立方体（仅保留有效样本）
    for i, feat in enumerate(feature_cols):
        # 初始化空 DataArray
        shap_da = xr.full_like(gw, np.nan, dtype=np.float32)
        
        # 按样本回填
        for idx, (_, row) in enumerate(df.iterrows()):
            t = row["time"]
            y = row["y"]
            x = row["x"]
            try:
                shap_da.loc[dict(time=t, y=y, x=x)] = shap_values[idx, i]
            except KeyError:
                continue  # 时间/坐标不匹配跳过
        
        # 保存
        out_path = Path(output_dir) / f"shap_{feat}.nc"
        shap_da.name = f"SHAP_{feat}"
        shap_da.to_netcdf(out_path)
        print(f"✅ 保存 {feat} 的空间 SHAP 图: {out_path}")

# 辅助函数（需从主脚本导入）
def add_lagged_features(df, var="NDVI", lags=[1, 2]):
    df = df.sort_values(["y", "x", "time"]).reset_index(drop=True)
    for lag in lags:
        df[f"{var}_lag{lag}"] = df.groupby(["y", "x"])[var].shift(lag)
    return df
