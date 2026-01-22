import numpy as np

def rmse(a, b):
    d = a - b
    return float(np.sqrt(np.mean(d * d)))

def rel_rmse(a, b, eps=1e-12):
    return float(np.linalg.norm((a - b).ravel()) / (np.linalg.norm(b.ravel()) + eps))

def bone_leakage_metric(LI_est, LB, thresh=None):
    """
    更稳健：只在 LB>0 的骨投影上取分位数，避免 thresh=0
    """
    LBv = LB.ravel()
    nz = LBv[LBv > 0]

    if nz.size == 0:
        return {'lb_thresh': 0.0, 'bone_mask_count': 0, 'falseI_mean': np.nan, 'falseI_abs_mean': np.nan}

    if thresh is None:
        # 取骨投影非零值的 20% 分位（相当于“确实穿过骨的一部分射线”）
        thresh = float(np.quantile(nz, 0.50))  # 或 0.70


    mask = LB > thresh
    return {
        'lb_thresh': float(thresh),
        'bone_mask_count': int(mask.sum()),
        'falseI_mean': float(LI_est[mask].mean()),
        'falseI_abs_mean': float(np.abs(LI_est[mask]).mean()),
    }

