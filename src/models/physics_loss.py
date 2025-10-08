# 地下水变化（ΔGW）通常具有 平滑性 和 质量守恒约束（如：不能剧烈震荡、趋势应与降水/蒸散发一致）。
# 我们可将这些先验知识融入损失函数。

# src/models/physics_loss.py
"""
Physics-guided custom objective for LightGBM.
Penalizes non-smooth predictions in time (e.g., high curvature in ΔGW).
Assumes input data is sorted by time and grouped by (lat, lon).
"""

import numpy as np
import pandas as pd

def physics_guided_objective(lam=0.1, group_col='group_id'):
    """
    Returns a LightGBM-compatible custom objective function.

    Parameters
    ----------
    lam : float
        Weight of physics loss (0 = pure MSE, 1 = pure physics).
    group_col : str
        Column name in dataset indicating spatial group (e.g., pixel ID).

    Returns
    -------
    objective : callable
        Function with signature (preds, train_data) -> (grad, hess)
    """
    def _objective(preds, train_data):
        y_true = train_data.get_label()
        df = train_data.construct().copy()  # LightGBM Dataset → pd.DataFrame
        df['pred'] = preds
        df['y_true'] = y_true

        # Ensure sorted by time within each group
        if 'time' in df.columns:
            df = df.sort_values([group_col, 'time'])
        else:
            # Fallback: assume already time-sorted globally
            pass

        # 1. MSE gradient & hessian
        grad_mse = preds - y_true
        hess_mse = np.ones_like(preds)

        # 2. Physics loss: penalize second-order time derivative (curvature)
        # Approximate d²y/dt² via finite difference
        pred_series = df['pred'].values
        group_ids = df[group_col].values if group_col in df.columns else np.zeros(len(df))

        grad_phys = np.zeros_like(preds)
        hess_phys = np.zeros_like(preds)

        # Compute curvature per group
        i = 0
        while i < len(pred_series):
            j = i
            while j < len(group_ids) and group_ids[j] == group_ids[i]:
                j += 1
            # Segment [i, j)
            seg_len = j - i
            if seg_len >= 3:
                # Second derivative: f[t] ≈ f[t-1] - 2f[t] + f[t+1]
                # Loss = sum (f[t-1] - 2f[t] + f[t+1])^2
                # Gradient w.r.t f[t]:
                #   t=0: -2*(f0 - 2f1 + f2)
                #   t=1: 2*(f0 - 2f1 + f2) - 2*(f1 - 2f2 + f3)
                #   ...
                seg_pred = pred_series[i:j]
                curvature = np.zeros(seg_len)
                for t in range(1, seg_len - 1):
                    curvature[t] = seg_pred[t-1] - 2*seg_pred[t] + seg_pred[t+1]

                # Gradient of sum(curvature^2)
                grad_seg = np.zeros(seg_len)
                for t in range(1, seg_len - 1):
                    grad_seg[t-1] += 2 * curvature[t]
                    grad_seg[t]   += -4 * curvature[t]
                    grad_seg[t+1] += 2 * curvature[t]

                grad_phys[i:j] = grad_seg
                hess_phys[i:j] = 4  # Approximate Hessian (constant for quadratic)

            i = j

        # Combine losses
        grad = (1 - lam) * grad_mse + lam * grad_phys
        hess = (1 - lam) * hess_mse + lam * hess_phys

        return grad, hess

    return _objective


def physics_eval_metric(lam=0.1, group_col='group_id'):
    """Optional: custom eval metric for monitoring."""
    def _eval(preds, train_data):
        y_true = train_data.get_label()
        mse = np.mean((preds - y_true) ** 2)

        # Simple smoothness penalty (for logging only)
        df = train_data.construct()
        df['pred'] = preds
        if 'time' in df.columns:
            df = df.sort_values([group_col, 'time'])
        pred_smooth = df['pred'].values
        smooth_penalty = np.mean(np.diff(np.diff(pred_smooth)) ** 2)

        total = (1 - lam) * mse + lam * smooth_penalty
        return 'physics_loss', total, False  # False = lower is better

    return _eval
