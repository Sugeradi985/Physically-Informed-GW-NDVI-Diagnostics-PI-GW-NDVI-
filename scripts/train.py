# scripts/train.py
"""
Physically-Informed Groundwater–Vegetation Diagnostic Model Training Script.

Features:
  - Auto lag selection via ACF or manual lag config
  - Human factors (irrigation, well density)
  - Physics-guided LightGBM (custom objective for smooth ΔGW)
  - SHAP interpretation saved as NetCDF
  - XGBoost fallback
"""

import os
import argparse
import xarray as xr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Model imports
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor

# Project modules
from src.data.load_data import load_gw, load_ndvi, load_climate, load_human_factors
from src.data.align import align_datasets
from src.features.lag_features import add_lagged_vars
from src.explain.shap_utils import save_shap_as_netcdf

# Optional: physics loss and auto-lag
try:
    from src.models.physics_loss import physics_guided_objective, physics_eval_metric
    PHYSICS_LOSS_AVAILABLE = True
except ImportError:
    PHYSICS_LOSS_AVAILABLE = False
    print("Warning: physics_loss.py not found. Physics-guided training disabled.")

try:
    from src.features.auto_lag_selection import suggest_lags_from_acf
    AUTO_LAG_AVAILABLE = True
except ImportError:
    AUTO_LAG_AVAILABLE = False
    print("Warning: auto_lag_selection.py not found. Auto-lag disabled.")


def get_model(args, y_train=None):
    """Return configured model."""
    if args.model == "xgboost":
        return XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
    elif args.model == "lightgbm":
        if args.physics_weight > 0 and not PHYSICS_LOSS_AVAILABLE:
            raise RuntimeError("Physics-guided loss requested but physics_loss.py not available.")
        
        if args.physics_weight > 0:
            objective = physics_guided_objective(lam=args.physics_weight, group_col='pixel_id')
            eval_metric = physics_eval_metric(lam=args.physics_weight, group_col='pixel_id')
        else:
            objective = 'regression'
            eval_metric = 'l2'

        return LGBMRegressor(
            n_estimators=1000,
            objective=objective,
            metric='custom' if args.physics_weight > 0 else 'l2',
            max_depth=8,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
    else:
        raise ValueError("model must be 'xgboost' or 'lightgbm'")


def main(args):
    print(f"🚀 Starting training at {pd.Timestamp.now()}")
    print(f"Project root: {args.ProjectPath}")
    
    # Step 1: Load raw data
    gw_da = load_gw(args.data_path)                     # xr.DataArray: delta_gw
    ndvi_da = load_ndvi(args.data_path)                 # xr.DataArray
    precip_da, temp_da = load_climate(args.data_path)   # xr.DataArrays

    # Step 2: Align all on common spatio-temporal grid
    ds = align_datasets(gw_da, ndvi_da, precip_da, temp_da)  # xr.Dataset

    # Step 3: Optionally load human factors
    if args.include_human:
        human_vars = load_human_factors(
            args.data_path,
            target_coords=ds[['lat', 'lon']],
            target_time=ds.time
        )
        for name, da in human_vars.items():
            ds[name] = da

    # Step 4: Determine lag configuration
    if args.auto_lag:
        if not AUTO_LAG_AVAILABLE:
            raise RuntimeError("--auto_lag requested but auto_lag_selection.py not found.")
        lag_config = {}
        auto_lag_vars = args.auto_lag_vars or ['NDVI']
        for var in auto_lag_vars:
            if var in ds.data_vars:
                lags = suggest_lags_from_acf(
                    ds[var],
                    max_lag=args.max_lag,
                    threshold=args.acf_threshold
                )
                lag_config[var] = lags
                print(f"✅ Auto-selected lags for {var}: {lags}")
            else:
                print(f"⚠️  Variable '{var}' not in dataset. Skipping.")
    else:
        lag_config = {}
        for var in args.lag_vars:
            if var in ds.data_vars:
                lag_config[var] = args.lag_steps
            else:
                print(f"⚠️  Variable '{var}' not in dataset. Skipping.")

    # Step 5: Add lagged features (exclude target 'delta_gw')
    predictors = ds.drop_vars('delta_gw')
    ds_lagged = add_lagged_vars(
        predictors,
        lag_config=lag_config,
        target_time=gw_da
    )
    ds_final = ds_lagged.assign(delta_gw=gw_da)

    # Step 6: Convert to DataFrame
    df = ds_final.to_dataframe()
    df = df.dropna()
    print(f"📊 Total samples after lag & dropna: {len(df)}")

    if len(df) == 0:
        raise RuntimeError("No valid samples! Check data alignment or lag steps.")

    # Add pixel_id for grouping (required by physics loss)
    df['pixel_id'] = (
        df.index.get_level_values('lat').astype(str) + '_' +
        df.index.get_level_values('lon').astype(str)
    )

    X = df.drop(columns=['delta_gw'])
    y = df['delta_gw']

    # Step 7: Train-test split (no shuffle for time series)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Step 8: Train model
    print(f"🧠 Training {args.model} (physics_weight={args.physics_weight})...")
    model = get_model(args, y_train.values)

    if args.model == "lightgbm" and args.physics_weight > 0:
        callbacks = [
            lgb.early_stopping(stopping_rounds=30, verbose=True),
            lgb.log_evaluation(period=20)
        ]
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=physics_eval_metric(lam=args.physics_weight, group_col='pixel_id'),
            callbacks=callbacks
        )
    else:
        model.fit(X_train, y_train)

    # Step 9: SHAP interpretation
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Reconstruct coordinates for SHAP
        test_index = X_test.index
        shap_da = xr.DataArray(
            shap_values,
            coords={
                'sample': np.arange(len(test_index)),
                'feature': X_test.columns
            },
            dims=['sample', 'feature']
        )
        coords_da = ds_final[['lat', 'lon', 'time']].sel(
            time=test_index.get_level_values('time'),
            lat=test_index.get_level_values('lat'),
            lon=test_index.get_level_values('lon')
        )
        shap_da = shap_da.assign_coords(
            time=('sample', coords_da.time.values),
            lat=('sample', coords_da.lat.values),
            lon=('sample', coords_da.lon.values)
        )

        output_path = os.path.join(args.output_dir, "shap_values.nc")
        save_shap_as_netcdf(shap_da, output_path)
        print(f"🎯 SHAP values saved to: {output_path}")

    except Exception as e:
        print(f"❌ SHAP failed: {e}")

    # Step 10: Save model
    model_path = os.path.join(args.output_dir, f"model_{args.model}.txt")
    if args.model == "xgboost":
        model.save_model(model_path)
    elif args.model == "lightgbm":
        model.booster_.save_model(model_path)
    print(f"💾 Model saved to: {model_path}")

    print("✅ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PI-GW-NDVI diagnostic model.")
    parser.add_argument("--ProjectPath", type=str, required=True, help="Root project path")
    parser.add_argument("--data_path", type=str, default=None, help="Path to input NetCDF data")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")

    # Model
    parser.add_argument("--model", type=str, default="lightgbm", choices=["xgboost", "lightgbm"])
    parser.add_argument("--physics_weight", type=float, default=0.1,
                        help="Weight of physics loss (0 = disabled, LightGBM only)")

    # Lag features
    parser.add_argument("--lag_vars", type=str, nargs="+", default=["NDVI"],
                        help="Variables to lag (manual mode)")
    parser.add_argument("--lag_steps", type=int, nargs="+", default=[1, 2],
                        help="Lag steps (e.g., 1 2 3)")

    # Auto-lag (overrides manual)
    parser.add_argument("--auto_lag", action="store_true", help="Auto-select lags using ACF")
    parser.add_argument("--auto_lag_vars", type=str, nargs="+", default=["NDVI"],
                        help="Variables for auto-lag selection")
    parser.add_argument("--max_lag", type=int, default=6, help="Max lag for ACF")
    parser.add_argument("--acf_threshold", type=float, default=0.2,
                        help="ACF significance threshold")

    # Human factors
    parser.add_argument("--include_human", action="store_true",
                        help="Include irrigation/well density")

    args = parser.parse_args()

    if args.data_path is None:
        args.data_path = os.path.join(args.ProjectPath, "data")
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
