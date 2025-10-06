import lightgbm as lgb
import numpy as np
import pandas as pd

def physics_informed_loss(y_true, y_pred, X, feat_names, lambda_phys=0.1):
    """
    自定义 LightGBM 损失：数据拟合 + 物理残差
    假设：ΔGW ≈ Precip - a*NDVI - b*Irrigation
    """
    y_pred = y_pred.reshape(-1)
    precip = X[:, feat_names.index("Precip")]
    ndvi = X[:, feat_names.index("NDVI")]
    irrig = X[:, feat_names.index("Irrigation")] if "Irrigation" in feat_names else np.zeros_like(precip)

    # 物理代理模型（系数可调）
    phys_pred = precip - 1.2 * ndvi - 0.8 * irrig
    phys_residual = y_pred - phys_pred

    # 梯度 & Hessian
    grad = -2 * (y_true - y_pred) + 2 * lambda_phys * phys_residual
    hess = 2 + 2 * lambda_phys

    return grad, hess

def train_phys_informed_lgbm(df, feature_cols, target="DeltaGW", lambda_phys=0.1):
    X = df[feature_cols].values
    y = df[target].values

    train_data = lgb.Dataset(
        X, y,
        free_raw_data=False,
        params={"verbose": -1}
    )

    model = lgb.train(
        params={
            "objective": "regression",
            "metric": "rmse",
            "verbose": -1
        },
        train_set=train_data,
        fobj=lambda y_true, y_pred: physics_informed_loss(
            y_true, y_pred, X, feature_cols, lambda_phys
        ),
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50)]
    )
    return model
