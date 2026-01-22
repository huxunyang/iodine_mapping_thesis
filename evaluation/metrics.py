import numpy as np

def rmse(a, b):
    d = a - b
    return float(np.sqrt(np.mean(d * d)))

def rel_rmse(a, b, eps=1e-12):
    return float(np.linalg.norm((a - b).ravel()) / (np.linalg.norm(b.ravel()) + eps))

def bone_leakage_metric(LI_est, LB, thresh=None):
    if thresh is None:
        thresh = float(np.quantile(LB, 0.90) * 0.10)
    mask = LB > thresh
    if mask.sum() == 0:
        return {'lb_thresh': thresh, 'bone_mask_count': 0, 'falseI_mean': np.nan, 'falseI_abs_mean': np.nan}
    return {
        'lb_thresh': thresh,
        'bone_mask_count': int(mask.sum()),
        'falseI_mean': float(LI_est[mask].mean()),
        'falseI_abs_mean': float(np.abs(LI_est[mask]).mean()),
    }
