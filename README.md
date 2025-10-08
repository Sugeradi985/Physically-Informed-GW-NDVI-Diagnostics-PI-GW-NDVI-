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

## 🚀 下一步建议

- 将此模板发布到 GitHub，并启用 **GitHub Pages** 展示示例结果图
- 添加 **Dockerfile**（参考知识库 [2] 的 CI 思路）
- 集成 **GRACE Mascon 下载脚本**（自动化数据获取）

需要我为你生成 **完整的 ZIP 项目包** 或 **GitHub 仓库初始化脚本** 吗？
