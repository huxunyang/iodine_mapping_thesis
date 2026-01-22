import numpy as np

def fit_effective_mu_2mat(W_gt, I_gt, Y, sample=200000, seed=0):
    """
    在图像域拟合有效系数：
        Y ≈ muW * W_gt + muI * I_gt
    返回 (muW, muI)
    """
    x1 = W_gt.ravel()
    x2 = I_gt.ravel()
    y  = Y.ravel()

    X = np.stack([x1, x2], axis=1)

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    m = min(sample, n)
    idx = rng.choice(n, size=m, replace=False)
    Xs = X[idx]
    ys = y[idx]

    coef, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    muW, muI = float(coef[0]), float(coef[1])
    return muW, muI

def decompose_WI_image(Y1, Y2, muW1, muI1, muW2, muI2, eps=1e-9):
    """
    逐体素闭式解：
        [Y1;Y2] = [[muW1 muI1],[muW2 muI2]] [W; I]
    输出：W_est, I_est（与 W_gt/I_gt 同单位/同尺度的“材料分量图”）
    """
    det = muW1 * muI2 - muW2 * muI1
    if abs(det) < eps:
        raise ValueError("Singular 2x2 system.")
    W = ( Y1 * muI2 - Y2 * muI1) / det
    I = (-Y1 * muW2 + Y2 * muW1) / det
    return W, I
