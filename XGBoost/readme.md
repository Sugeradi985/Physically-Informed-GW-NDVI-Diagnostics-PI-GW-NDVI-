要构建第一个 XGBoost 诊断模型，用于验证 ΔGW（地下水异常）与 NDVI（植被指数）之间的相关性，我们可以将问题建模为：

以 NDVI 为输入特征，预测 ΔGW 的值（回归任务）

若模型具有显著预测能力（如 R² > 0.3 或 RMSE 明显低于标准差），则说明二者存在可学习的非线性相关性。

✅ 步骤概览
准备对齐的时空数据（ΔGW + NDVI）
构建样本：每个像元 × 时间点 → 一个样本
训练 XGBoost 回归模型
评估模型性能 & 特征重要性
可视化预测 vs 真实值
可选：空间制图预测残差或 R²

📦 所需依赖
```bash
pip install xgboost scikit-learn matplotlib seaborn
```

🧠 完整代码：xgboost_gw_ndvi_diagnosis.py


### 使用示例
回归任务（带气候变量 + 滞后）
```bash
python xgboost_gw_diagnosis_enhanced.py \
  --delta_gw "results/delta_gw.nc" \
  --ndvi "data/ndvi.nc" \
  --precip "data/precip.nc" \
  --temp "data/temp.nc" \
  --task regression \
  --shap-summary "figures/shap_summary_reg.png" \
  --shap-dependence "figures/shap_dependence_ndvi_precip.png"
```

分类任务（ΔGW 上升/下降）
```bash
python xgboost_gw_diagnosis_enhanced.py \
  --delta_gw "results/delta_gw.nc" \
  --ndvi "data/ndvi.nc" \
  --precip "data/precip.nc" \
  --task classification \
  --shap-summary "figures/shap_summary_cls.png"
```

📊 输出解读
输出	                                    说明
SHAP Summary (Beeswarm)	      显示哪些特征最重要（如 NDVI_lag1 > NDVI 表明植被响应有滞后）
SHAP Dependence Plot	        揭示非线性关系（如高 NDVI 时降水对 ΔGW 影响更强）
分类准确率 > 60%	              表明自然系统中植被-水文存在可预测的耦合机制
R² 提升（加入滞后/气候）	      证明时序动态与外部驱动的重要性


📦 依赖安装
```bash
pip install xgboost shap scikit-learn xarray pandas matplotlib seaborn
```
💡 提示：首次运行 shap 可能较慢（需编译），建议使用 conda install -c conda-forge shap 加速。


