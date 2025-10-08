import xarray as xr
import rioxarray  # for CRS handling
import pandas as pd
from pathlib import Path

def load_and_align(
    delta_gw: str,
    ndvi: str,
    precip: str,
    temp: str,
    irrigation: str = None,
    target_resolution: float = 0.25  # degrees (~25km)
):
    """
    Load and temporally/spatially align GRACE, NDVI, climate, and irrigation data.
    
    Returns:
        aligned_data (dict): xarray DataArrays aligned on common time/space grid.
    """
    # Load GRACE DeltaGW
    gw = xr.open_dataset(delta_gw).rename({'lwe_thickness': 'DeltaGW'})
    gw = gw['DeltaGW'].resample(time='1MS').mean()  # Monthly

    # Load NDVI (e.g., from MODIS)
    ndvi_da = xr.open_dataset(ndvi)['NDVI'].resample(time='1MS').mean()

    # Load climate
    precip_da = xr.open_dataset(precip)['Precip'].resample(time='1MS').mean()
    temp_da = xr.open_dataset(temp)['Temp'].resample(time='1MS').mean()

    # Align spatial resolution (coarsen to GRACE scale)
    ndvi_coarse = ndvi_da.coarsen(x=int(target_resolution/0.005), y=int(target_resolution/0.005), boundary='trim').mean()
    precip_coarse = precip_da.coarsen(x=int(target_resolution/0.05), y=int(target_resolution/0.05), boundary='trim').mean()
    temp_coarse = temp_da.coarsen(x=int(target_resolution/0.1), y=int(target_resolution/0.1), boundary='trim').mean()

    # Reindex to common time
    common_time = gw.time
    aligned = {
        "DeltaGW": gw.reindex(time=common_time, method='nearest'),
        "NDVI": ndvi_coarse.reindex(time=common_time, method='nearest'),
        "Precip": precip_coarse.reindex(time=common_time, method='nearest'),
        "Temp": temp_coarse.reindex(time=common_time, method='nearest')
    }

    if irrigation:
        irrig = xr.open_dataset(irrigation)['irrigation_fraction']
        aligned["Irrigation"] = irrig

    return aligned
