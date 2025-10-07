#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版 XGBoost 诊断模型：
- 滞后 NDVI（t, t-1, t-2）
- 气候协变量（降水、温度）
- 支持回归 & 分类任务
- SHAP 可解释性分析（全局 + 局部）
"""

import xarray as xr
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP（必须安装）
import shap

# ======================
# 工具函数
# ======================

def load_and_align(delta_gw_path, ndvi_path, precip_path=None, temp_path=None):
    """加载所有变量并时空对齐"""
    gw = xr.open_dataarray(delta_gw_path)
    ndvi = xr.open_dataarray(ndvi_path)

    # 标准化坐标
    for da in [gw, ndvi]:
        if "lat" in da.dims: da = da.rename({"lat": "y", "lon": "x"})
        if "latitude" in da.dims: da = da.rename({"latitude": "y", "longitude": "x"})

    datasets = [gw, ndvi]
    names = ["DeltaGW", "NDVI"]

    if precip_path:
        precip = xr.open_dataarray(precip_path)
        if "lat" in precip.dims: precip = precip.rename({"lat": "y", "lon": "x"})
        datasets.append(precip)
        names.append("Precip")
    if temp_path:
        temp = xr.open_dataarray(temp_path)
        if "lat" in temp.dims: temp = temp.rename({"lat": "y", "lon": "x"})
        datasets.append(temp)
        names.append("Temp")

    # 时间交集
    common_time = datasets[0].time
    for da in datasets[1:]:
        common_time = np.intersect1d(common_time, da.time)
    
    aligned = [da.sel(time=common_time) for da in datasets]
    return dict(zip(names, aligned))


def add_lagged_features(df, var="NDVI", lags=[1, 2]):
    """为 DataFrame 添加滞后特征（按像元分组）"""
    df = df.copy()
    df = df.sort_values(["y", "x", "time"]).reset_index(drop=True)
    
    for lag in lags:
        df[f"{var}_lag{lag}"] = df.groupby(["y", "x"])[var].shift(lag)
    return df


def build_features(gw_da, ndvi_da, precip_da=None, temp_da=None, lags=[1, 2], task="regression"):
    """构建完整特征集 + 标签"""
    # 展平
    gw_flat = gw_da.stack(sample=("time", "y", "x"))
    ndvi_flat = ndvi_da.stack(sample=("time", "y", "x"))

    df = pd.DataFrame({
        "time": gw_flat.time.values,
        "y": gw_flat.y.values,
        "x": gw_flat.x.values,
        "DeltaGW": gw_flat.values,
        "NDVI": ndvi_flat.values
    })

    if precip_da is not None:
        precip_flat = precip_da.stack(sample=("time", "y", "x"))
        df["Precip"] = precip_flat.values
    if temp_da is not None:
        temp_flat = temp_da.stack(sample=("time", "y", "x"))
        df["Temp"] = temp_flat.values

    df = df.dropna().reset_index(drop=True)
    df = add_lagged_features(df, "NDVI", lags)

    # 删除仍含 NaN 的行（因滞后）
    df = df.dropna().reset_index(drop=True)

    # 构建标签
    if task == "classification":
        # ΔGW 上升（+1） vs 下降（-1）
        df["DeltaGW_trend"] = np.sign(df["DeltaGW"].diff())
        # 移除首行 NaN
        df = df.dropna(subset=["DeltaGW_trend"]).copy()
        df["DeltaGW_trend"] = df["DeltaGW_trend"].astype(int)
        # 合并为 0/1（可选）
        df["label"] = (df["DeltaGW_trend"] > 0).astype(int)
    else:
        df["label"] = df["DeltaGW"]

    # 特征列
    feature_cols = ["NDVI"] + [f"NDVI_lag{lag}" for lag in lags]
    if "Precip" in df.columns:
        feature_cols.append("Precip")
    if "Temp" in df.columns:
        feature_cols.append("Temp")

    return df[feature_cols + ["label", "time", "y", "x"]], feature_cols


def train_model(df, feature_cols, task="regression", random_state=42):
    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y if task=="classification" else None
    )

    if task == "regression":
        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"📊 回归性能：R² = {r2_score(y_test, y_pred):.3f} | RMSE = {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    else:
        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss"
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"📊 分类性能：Accuracy = {accuracy_score(y_test, y_pred):.3f}")
        print(classification_report(y_test, y_pred, target_names=["下降", "上升"]))

    return model, X_train, X_test, y_train, y_test, y_pred


def plot_shap_summary(model, X_test, feature_cols, output_plot=None):
    """SHAP 全局解释（Beeswarm 图）"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 6))
    if len(shap_values.shape) == 2:  # regression or binary classification
        shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
    else:  # multiclass
        shap.summary_plot(shap_values[1], X_test, feature_names=feature_cols, show=False)

    if output_plot:
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"🔍 SHAP 全局解释图已保存：{output_plot}")
    else:
        plt.show()


def plot_shap_dependence(model, X_test, feature_cols, main_feature="NDVI", interaction_feature="Precip", output_plot=None):
    """SHAP 依赖图（展示非线性 & 交互效应）"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    idx_main = feature_cols.index(main_feature)
    if interaction_feature in feature_cols:
        idx_inter = feature_cols.index(interaction_feature)
        interaction_index = idx_inter
    else:
        interaction_index = None

    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        idx_main,
        shap_values,
        X_test,
        feature_names=feature_cols,
        interaction_index=interaction_index,
        show=False
    )
    if output_plot:
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"🔍 SHAP 依赖图已保存：{output_plot}")
    else:
        plt.show()


# ======================
# 主函数
# ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta_gw", required=True)
    parser.add_argument("--ndvi", required=True)
    parser.add_argument("--precip", default=None)
    parser.add_argument("--temp", default=None)
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--shap-summary", default=None, help="SHAP 全局图路径")
    parser.add_argument("--shap-dependence", default=None, help="SHAP 依赖图路径")
    args = parser.parse_args()

    # 1. 加载数据
    data = load_and_align(
        args.delta_gw, args.ndvi,
        precip_path=args.precip,
        temp_path=args.temp
    )

    # 2. 构建特征
    df, feature_cols = build_features(
        data["DeltaGW"], data["NDVI"],
        precip_da=data.get("Precip"),
        temp_da=data.get("Temp"),
        lags=[1, 2],
        task=args.task
    )

    print(f"✅ 特征列：{feature_cols}")
    print(f"📊 样本量：{len(df)}")

    # 3. 训练模型
    model, X_train, X_test, y_train, y_test, y_pred = train_model(
        df, feature_cols, task=args.task
    )

    # 4. SHAP 解释
    if args.shap_summary:
        plot_shap_summary(model, X_test, feature_cols, args.shap_summary)
    if args.shap_dependence:
        plot_shap_dependence(
            model, X_test, feature_cols,
            main_feature="NDVI",
            interaction_feature="Precip" if "Precip" in feature_cols else None,
            output_plot=args.shap_dependence
        )

    # 5. 结论提示
    if args.task == "regression":
        r2 = r2_score(y_test, y_pred)
        if r2 > 0.2:
            print("\n✅ NDVI（含滞后）与气候变量对 ΔGW 有显著预测能力。")
        else:
            print("\n⚠️ 预测能力有限，可能受人为抽水等未观测因素主导。")
    else:
        acc = accuracy_score(y_test, y_pred)
        if acc > 0.6:
            print("\n✅ 植被与气候信号可有效指示地下水变化方向。")
        else:
            print("\n⚠️ 地下水变化方向难以仅由自然变量预测。")


if __name__ == "__main__":
    main()
