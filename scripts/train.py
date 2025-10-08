# scripts/train.py
"""
Main training script for Physically-Informed Groundwater-Vegetation Diagnostics.
Supports:
  - Lagged features (e.g., NDVI_lag1, NDVI_lag2)
  - Human factors (irrigation, well_density)
  - Physics-guided LightGBM or standard XGBoost
  - SHAP interpretation saved as NetCDF
"""

import os
import argparse
import xarray as xr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Model imports
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Project modules
from src.data.load_data import load_gw, load_ndvi, load_climate, load_human_factors
from src.data.align import align_datasets
from src.features.lag_features import add_lagged_vars
from src.explain.shap_utils import save_shap_as_netcdf  # Assume this exists in your repo


def get_model(model_name: str, physics_weight: float = 0.0, y_train: np.ndarray = None):
    """Return configured model."""
    if model_name == "xgboost":
        return XGBRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    elif model_name == "lightgbm":
        # Optional: integrate physics-guided loss if implemented
        # For now, use standard LGBM; you can replace with custom objective
        return LGBMRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    else:
        raise ValueError("model must be 'xgboost' or 'lightgbm'")


def main(args):
    print(f"Loading data from: {args.data_path}")
    
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

    # Step 4: Build lag configuration
    lag_config = {}
    for var in args.lag_vars:
        if var in ds.data_vars:
            lag_config[var] = args.lag_steps
        else:
            print(f"Warning: Variable '{var}' not found. Skipping lag creation.")

    # Step 5: Add lagged features (exclude target 'delta_gw')
    predictors = ds.drop_vars('delta_gw')
    ds_lagged = add_lagged_vars(
        predictors,
        lag_config=lag_config,
        target_time=gw_da  # align with delta_gw time
    )

    # Re-attach target
    ds_final = ds_lagged.assign(delta_gw=gw_da)

    # Step 6: Convert to DataFrame and drop NaN (from lags)
    df = ds_final.to_dataframe()
    df_clean = df.dropna()
    print(f"Total samples after lag & dropna: {len(df_clean)}")

    if len(df_clean) == 0:
        raise RuntimeError("No valid samples after lag processing. Check time alignment or lag steps.")

    X = df_clean.drop(columns=['delta_gw'])
    y = df_clean['delta_gw']

    # Step 7: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # time-series: no shuffle
    )

    # Step 8: Train model
    print(f"Training {args.model} model...")
    model = get_model(args.model, physics_weight=args.physics_weight, y_train=y_train.values)
    model.fit(X_train, y_train)

    # Step 9: SHAP interpretation
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Reshape SHAP to original space-time grid for saving
        # Reconstruct xarray structure from test index
        test_index = X_test.index  # MultiIndex: (time, lat, lon)
        shap_da = xr.DataArray(
            shap_values,
            coords={
                'sample': np.arange(len(test_index)),
                'feature': X_test.columns
            },
            dims=['sample', 'feature']
        )
        # Attach original coordinates
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

        # Save as NetCDF (reuse your existing utility)
        output_path = os.path.join(args.output_dir, "shap_values.nc")
        save_shap_as_netcdf(shap_da, output_path)
        print(f"SHAP values saved to: {output_path}")

    except Exception as e:
        print(f"SHAP failed: {e}")

    # Step 10: Save model (optional)
    model_path = os.path.join(args.output_dir, f"model_{args.model}.json")
    if args.model == "xgboost":
        model.save_model(model_path)
    elif args.model == "lightgbm":
        model.booster_.save_model(model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PI-GW-NDVI diagnostic model.")
    parser.add_argument("--ProjectPath", type=str, required=True, help="Root project path")
    parser.add_argument("--data_path", type=str, default=None, help="Path to input NetCDF data")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")

    # Model options
    parser.add_argument("--model", type=str, default="lightgbm", choices=["xgboost", "lightgbm"])
    parser.add_argument("--physics_weight", type=float, default=0.1, help="Physics loss weight (LightGBM only)")

    # Lag features
    parser.add_argument("--lag_vars", type=str, nargs="+", default=["NDVI"], help="Variables to lag")
    parser.add_argument("--lag_steps", type=int, nargs="+", default=[1, 2], help="Lag steps (e.g., 1 2 3)")

    # Human factors
    parser.add_argument("--include_human", action="store_true", help="Include irrigation/well density")

    args = parser.parse_args()

    # Set default data_path if not provided
    if args.data_path is None:
        args.data_path = os.path.join(args.ProjectPath, "data")

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
