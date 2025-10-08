import numpy as np
import pandas as pd
import xarray as xr

def build_features(aligned_data, lags=[1, 2]):
    """
    Flatten spatial-temporal data into DataFrame with lagged features.
    
    Returns:
        df (pd.DataFrame): flattened feature matrix
        feature_cols (list): column names
    """
    gw = aligned_data["DeltaGW"]
    df_list = []

    for var_name in ["NDVI", "Precip", "Temp", "Irrigation"]:
        if var_name in aligned_data:
            da = aligned_data[var_name]
            # Ensure same spatial grid
            if da.shape != gw.shape:
                da = da.interp_like(gw, method='nearest')
            df_list.append(da.to_dataframe(name=var_name).reset_index())

    # Merge all
    df = df_list[0]
    for d in df_list[1:]:
        df = pd.merge(df, d, on=['time', 'y', 'x'], how='inner')

    # Add lags
    feature_cols = ["NDVI", "Precip", "Temp"]
    if "Irrigation" in df.columns:
        feature_cols.append("Irrigation")

    for lag in lags:
        for col in feature_cols:
            df[f"{col}_lag{lag}"] = df.groupby(['y', 'x'])[col].shift(lag)
        feature_cols.extend([f"{col}_lag{lag}" for col in ["NDVI", "Precip"]])

    df = df.dropna().reset_index(drop=True)
    return df, feature_cols
