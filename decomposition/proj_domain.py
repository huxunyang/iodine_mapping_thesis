import numpy as np

def fit_mu_coeffs(LW, LI, LB, LT, pE, sample=200000, seed=0):
    X = np.stack([LW.ravel(), LI.ravel(), LB.ravel(), LT.ravel()], axis=1)
    y = pE.ravel()

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    m = min(sample, n)
    idx = rng.choice(n, size=m, replace=False)

    Xs = X[idx]
    ys = y[idx]
    coef, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    return coef  # [muW, muI, muB, muT]

def decompose_WI(p1, p2, muW1, muI1, muW2, muI2, eps=1e-9):
    det = muW1 * muI2 - muW2 * muI1
    if abs(det) < eps:
        raise ValueError('Singular 2x2 system for this energy pair.')
    LW = ( p1 * muI2 - p2 * muI1) / det
    LI = (-p1 * muW2 + p2 * muW1) / det
    return LW, LI
