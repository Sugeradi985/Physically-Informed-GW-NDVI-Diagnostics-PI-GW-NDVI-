import xarray as xr
import numpy as np
import pandas as pd
import shap
from pathlib import Path

def generate_spatial_shap_maps(model, aligned_data, feature_cols, output_dir="output/shap_maps/"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 构建样本（含时空索引）
    gw = aligned_data["DeltaGW"]
    df = pd.DataFrame({
        "time": np.tile(gw.time.values, gw.y.size * gw.x.size),
        "y": np.tile(gw.y.values, gw.time.size * gw.x.size),
        "x": np.tile(np.repeat(gw.x.values, gw.y.size), gw.time.size),
        "DeltaGW": gw.values.flatten()
    })

    # 添加其他变量
    for var in ["NDVI", "Precip", "Temp", "Irrigation"]:
        if var in aligned_data:
            df[var] = aligned_data[var].values.flatten()

    df = df.dropna().reset_index(drop=True)

    X = df[feature_cols].values
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    # 重建空间立方体
    for i, feat in enumerate(feature_cols):
        shap_da = xr.full_like(gw, np.nan, dtype=np.float32)
        for idx, row in df.iterrows():
            try:
                shap_da.loc[dict(time=row["time"], y=row["y"], x=row["x"])] = shap_vals[idx, i]
            except KeyError:
                continue
        shap_da.name = f"SHAP_{feat}"
        shap_da.to_netcdf(f"{output_dir}/shap_{feat}.nc")
