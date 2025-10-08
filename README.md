# PI-GW-NDVI: Physically-Informed Groundwater-Vegetation Diagnostics

```markdown
A reproducible framework to:
- Avoid pure data-driven pitfalls via physics-informed ML
- Align GRACE (150km) with MODIS/Sentinel (10m–500m) via spatial aggregation
- Separate climate vs human drivers using irrigation & pumping data
- Interpret spatial heterogeneity via pixel-wise SHAP

## Quick Start
```
1. **Export climate data** from GEE:
   ```js
   // Run gee/export_climate_covariates.js in GEE Code Editor


Prepare data in data/raw/ and data/external/

Run analysis:

```bash
conda env create -f environment.yml
jupyter notebook notebooks/gw_ndvi_diagnosis.ipynb
```

Place data:
data/raw/grace_delta_gw.nc
data/raw/ndvi.nc
data/raw/Precip.nc, Temp.nc
data/external/gmia_irrigation.nc (optional)

Setup environment:
```bash
conda env create -f environment.yml
conda activate pi-gw-ndvi
```

Run analysis:
```bash
jupyter notebook notebooks/gw_ndvi_diagnosis.ipynb
```

Outputs
output/shap_maps/shap_NDVI.nc
output/shap_maps/shap_Irrigation.nc
Model interpretability plots
Citation
If you use this template, please cite:



---

## ✅ 项目优势总结

| 需求 | 实现方式 |
|------|--------|
| **避免纯黑箱** | LightGBM + 物理残差损失函数 |
| **尺度匹配** | GRACE 网格为单元，聚合高分辨率变量 |
| **人类活动分离** | 引入 `Irrigation`, `Well_density` 等协变量 |
| **可解释性** | 像元级 SHAP 贡献图（NetCDF 输出） |
| **可复现性** | GitHub 模板 + Jupyter + 环境文件 |

---

滞后阶数选择建议：

干旱区：植被对地下水响应慢，可设 lag = [2, 3, 4]（月）
湿润区：响应快，lag = [0, 1, 2]
可通过 交叉验证 AUC / R² vs lag 阶数 自动选择最优窗口
参考 ADL 模型思想（知识库 [2]）：分布滞后部分捕捉解释变量（NDVI）的动态影响
