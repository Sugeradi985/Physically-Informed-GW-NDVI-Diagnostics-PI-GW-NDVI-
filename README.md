# Physically-Informed-GW-NDVI-Diagnostics-PI-GW-NDVI-

# PI-GW-NDVI: Physically-Informed Groundwater-Vegetation Diagnostics

A reproducible framework to:
- Avoid pure data-driven pitfalls via physics-informed ML
- Align GRACE (150km) with MODIS/Sentinel (10m–500m) via spatial aggregation
- Separate climate vs human drivers using irrigation & pumping data
- Interpret spatial heterogeneity via pixel-wise SHAP

## Quick Start

1. **Export climate data** from GEE:
   ```js
   // Run gee/export_climate_covariates.js in GEE Code Editor

Prepare data in data/raw/ and data/external/
Run analysis:

```bash
conda env create -f environment.yml
jupyter notebook notebooks/gw_ndvi_diagnosis.ipynb
```
