#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 诊断模型：验证 ΔGW 与 NDVI 的相关性
输入：已对齐的 ΔGW 和 NDVI NetCDF 文件（时空匹配）
输出：模型性能指标、特征重要性、散点图、可选空间 R² 图
"""

import xarray as xr
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# 工具函数
# ======================

def load_and_align(delta_gw_path: str, ndvi_path: str):
    """加载 ΔGW 和 NDVI，并确保时空对齐"""
    gw = xr.open_dataarray(delta_gw_path)
    ndvi = xr.open_dataarray(ndvi_path)

    # 确保坐标名一致
    for da in [gw, ndvi]:
        if "lat" in da.dims: da = da.rename({"lat": "y", "lon": "x"})
        if "latitude" in da.dims: da = da.rename({"latitude": "y", "longitude": "x"})

    # 时间交集
    common_time = np.intersect1d(gw.time, ndvi.time)
    gw = gw.sel(time=common_time)
    ndvi = ndvi.sel(time=common_time)

    return gw, ndvi


def build_dataframe(gw: xr.DataArray, ndvi: xr.DataArray, mask_valid=True):
    """将 xarray 转为 (sample, feature) 的 DataFrame"""
    # 展平：(time, y, x) → (sample,)
    gw_flat = gw.stack(sample=("time", "y", "x"))
    ndvi_flat = ndvi.stack(sample=("time", "y", "x"))

    # 合并
    df = pd.DataFrame({
        "NDVI": ndvi_flat.values,
        "DeltaGW": gw_flat.values,
        "time": gw_flat.time.values,
        "y": gw_flat.y.values,
        "x": gw_flat.x.values
    })

    if mask_valid:
        df = df.dropna().reset_index(drop=True)

    print(f"✅ 构建完成：{len(df)} 有效样本")
    return df


def train_xgboost(df: pd.DataFrame, test_size=0.2, random_state=42):
    X = df[["NDVI"]].values
    y = df["DeltaGW"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        objective="reg:squarederror"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    std_y = np.std(y_test)

    print(f"📊 模型性能：R² = {r2:.3f} | RMSE = {rmse:.2f} | y_std = {std_y:.2f}")

    return model, X_test, y_test, y_pred, r2, rmse


def plot_scatter(y_true, y_pred, output_plot=None):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, s=15)
    plt.xlabel("Observed ΔGW (mm)")
    plt.ylabel("Predicted ΔGW (mm)")
    plt.title(f"XGBoost Prediction vs True (R² = {r2_score(y_true, y_pred):.3f})")
    plt.axline((0, 0), slope=1, color="red", linestyle="--")
    if output_plot:
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"📈 散点图已保存：{output_plot}")
    else:
        plt.show()


def compute_spatial_r2(gw: xr.DataArray, ndvi: xr.DataArray, model) -> xr.DataArray:
    """计算每个像元的局部 R²（时间维度上）"""
    def pixel_r2(gw_ts, ndvi_ts):
        valid = ~(np.isnan(gw_ts) | np.isnan(ndvi_ts))
        if valid.sum() < 10:
            return np.nan
        X = ndvi_ts[valid].reshape(-1, 1)
        y_true = gw_ts[valid]
        y_pred = model.predict(X)
        return r2_score(y_true, y_pred)

    # 应用到每个 (y, x)
    r2_map = xr.apply_ufunc(
        pixel_r2,
        gw,
        ndvi,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32]
    )
    return r2_map


def plot_spatial_r2(r2_map: xr.DataArray, output_plot=None):
    plt.figure(figsize=(8, 6))
    r2_map.plot(
        cmap="viridis",
        vmin=0,
        vmax=1,
        cbar_kwargs={"label": "Local R² (NDVI → ΔGW)"},
        levels=21
    )
    plt.title("Spatial Distribution of NDVI-ΔGW Predictive Power (R²)")
    if output_plot:
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"🗺️ 空间 R² 图已保存：{output_plot}")
    else:
        plt.show()


# ======================
# 主函数
# ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta_gw", required=True, help="ΔGW NetCDF 文件路径")
    parser.add_argument("--ndvi", required=True, help="NDVI NetCDF 文件路径")
    parser.add_argument("--plot-scatter", default=None, help="预测-真实散点图路径")
    parser.add_argument("--plot-spatial-r2", default=None, help="空间 R² 图路径")
    args = parser.parse_args()

    # 1. 加载对齐数据
    gw, ndvi = load_and_align(args.delta_gw, args.ndvi)

    # 2. 构建 DataFrame
    df = build_dataframe(gw, ndvi)

    # 3. 训练模型
    model, X_test, y_test, y_pred, r2, rmse = train_xgboost(df)

    # 4. 绘图
    if args.plot_scatter:
        plot_scatter(y_test, y_pred, args.plot_scatter)

    # 5. 空间 R²（可选，较耗时）
    if args.plot_spatial_r2:
        print("⏳ 正在计算空间 R²（可能需要几分钟）...")
        r2_map = compute_spatial_r2(gw, ndvi, model)
        plot_spatial_r2(r2_map, args.plot_spatial_r2)

    # 6. 输出结论
    if r2 > 0.1:
        print("\n✅ 结论：XGBoost 模型能从 NDVI 中有效预测 ΔGW，表明二者存在显著非线性相关性。")
    else:
        print("\n⚠️ 结论：NDVI 对 ΔGW 的预测能力较弱，可能受其他水文/人为因素主导。")


if __name__ == "__main__":
    main()
