import lightgbm as lgb
import numpy as np

def physics_informed_loss(y_true, y_pred, X, feat_names, lambda_phys=0.1):
    """
    Custom loss: y_pred ≈ Precip - α·NDVI - β·Irrigation
    """
    y_pred = y_pred.reshape(-1)
    precip = X[:, feat_names.index("Precip")]
    ndvi = X[:, feat_names.index("NDVI")]
    irrig = X[:, feat_names.index("Irrigation")] if "Irrigation" in feat_names else np.zeros_like(precip)
    
    # Physical constraint: ΔGW ≈ P - ET ≈ P - (a·NDVI + b·Irrig)
    phys_pred = precip - 1.2 * ndvi - 0.8 * irrig
    phys_residual = y_pred - phys_pred

    grad = -2 * (y_true - y_pred) + 2 * lambda_phys * phys_residual
    hess = 2 + 2 * lambda_phys
    return grad, hess

def train_phys_informed_lgbm(df, feature_cols, target="DeltaGW", lambda_phys=0.1):
    X = df[feature_cols].values
    y = df[target].values
    train_data = lgb.Dataset(X, y, free_raw_data=False)
    
    model = lgb.train(
        params={
            "objective": "regression",
            "metric": "rmse",
            "verbose": -1,
            "seed": 42
        },
        train_set=train_data,
        fobj=lambda y_true, y_pred: physics_informed_loss(y_true, y_pred, X, feature_cols, lambda_phys),
        num_boost_round=500,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    return model
