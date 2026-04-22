import json
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt

import config as C


# -----------------------------
# DICOM utilities
# -----------------------------
def dcm_to_hu(ds):
    px = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return px * slope + intercept


def load_series_slices(folder: Path):
    files = [p for p in folder.rglob("*") if p.is_file()]
    dss = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
            if hasattr(ds, "PixelData"):
                dss.append(ds)
        except Exception:
            pass

    if not dss:
        raise RuntimeError(f"No DICOM slices found in {folder}")

    def sort_key(ds):
        if hasattr(ds, "ImagePositionPatient"):
            try:
                return float(ds.ImagePositionPatient[2])
            except Exception:
                pass
        if hasattr(ds, "InstanceNumber"):
            try:
                return float(ds.InstanceNumber)
            except Exception:
                pass
        return 0.0

    dss.sort(key=sort_key)
    return dss


def pick_slice_index(dss1, dss2, idx):
    idx = int(idx)
    idx = max(0, min(idx, min(len(dss1), len(dss2)) - 1))
    return idx


def circular_mask(h, w, x, y, r):
    yy, xx = np.ogrid[:h, :w]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2


def ensure_dir(p: Path):
    p.mkdir(exist_ok=True, parents=True)
    return p


# -----------------------------
# ROI table (HU)
# -----------------------------
def compute_roi_hu_table(roiconf):
    """
    Return df with columns:
      sid, kvp, rep, c0, c50, ...
    Mean HU averaged over slice_range.
    """
    s0, s1 = roiconf.get("slice_range", list(C.SLICE_RANGE))
    rois = roiconf["rois"]

    rows = []
    for sid, folder in C.SERIES.items():
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"{sid}: series folder not found: {folder}")

        dss = load_series_slices(folder)
        # average over slice_range
        s0i = max(0, int(s0))
        s1i = min(len(dss) - 1, int(s1))
        if s0i > s1i:
            raise ValueError(f"Invalid slice_range {s0}-{s1} for {sid} with n={len(dss)}")

        per_slice = []
        for ds in dss[s0i : s1i + 1]:
            hu = dcm_to_hu(ds)
            h, w = hu.shape
            vals = {}
            for r in rois:
                m = circular_mask(h, w, r["x"], r["y"], r["r_px"])
                vals[f"c{int(r['conc_mgml'])}"] = float(np.mean(hu[m]))
            per_slice.append(vals)

        keys = per_slice[0].keys()
        avg = {k: float(np.mean([d[k] for d in per_slice])) for k in keys}

        kvp = int(sid.split("_")[0])
        rep = int(sid.split("_")[1])
        rows.append({"sid": sid, "kvp": kvp, "rep": rep, **avg})

    return pd.DataFrame(rows).sort_values(["kvp", "rep"]).reset_index(drop=True)


# -----------------------------
# Material decomposition calibration (alpha/beta)
# -----------------------------
def estimate_alpha_beta(df, E, water_c=0, ref_c=200):
    """
    HU_E ≈ alpha_E * w + beta_E * c
    Use water ROI (c=0) -> alpha_E ~= HU_water
    Use reference iodine ROI (c=ref_c) -> beta_E = (HU_ref - HU_water)/ref_c
    Estimate from all reps and take mean (stable).
    """
    d = df[df["kvp"] == E].copy()
    if d.empty:
        raise RuntimeError(f"No rows for kVp={E} in ROI table")

    col_w = f"c{int(water_c)}"
    col_r = f"c{int(ref_c)}"
    if col_w not in d.columns or col_r not in d.columns:
        raise RuntimeError(f"Missing ROI columns for E={E}: need {col_w} and {col_r}")

    hu_w = d[col_w].to_numpy(np.float64)
    hu_r = d[col_r].to_numpy(np.float64)

    alpha = float(np.mean(hu_w))
    beta = float(np.mean((hu_r - hu_w) / float(ref_c)))
    return alpha, beta


def solve_2x2(alpha1, beta1, alpha2, beta2, y1, y2, eps=1e-12):
    """
    [y1;y2] = [[alpha1 beta1],[alpha2 beta2]] [w;c]
    """
    det = alpha1 * beta2 - alpha2 * beta1
    if abs(det) < eps:
        raise ValueError("Singular 2x2 system (det ~ 0).")
    w = ( y1 * beta2 - y2 * beta1) / det
    c = (-y1 * alpha2 + y2 * alpha1) / det
    return w, c, det


def fit_linear_calibration_from_rois(df, E1, E2, alpha1, beta1, alpha2, beta2, concs):
    """
    Use ROI means to compute c_raw per ROI then fit:
        c_true = s*c_raw + t
    Align by rep to avoid mismatch.
    """
    d1 = df[df["kvp"] == E1][["rep"] + [f"c{c}" for c in concs]].copy()
    d2 = df[df["kvp"] == E2][["rep"] + [f"c{c}" for c in concs]].copy()
    m = d1.merge(d2, on="rep", suffixes=(f"_{E1}", f"_{E2}"))
    if m.empty:
        raise RuntimeError(f"Empty merge for pair {E1}/{E2}. Check reps.")

    xs, ys = [], []
    for _, row in m.iterrows():
        for ctrue in concs:
            y1 = float(row[f"c{ctrue}_{E1}"])
            y2 = float(row[f"c{ctrue}_{E2}"])
            _, c_raw, _ = solve_2x2(alpha1, beta1, alpha2, beta2, y1, y2)
            xs.append(float(c_raw))
            ys.append(float(ctrue))

    X = np.asarray(xs, np.float64)
    Y = np.asarray(ys, np.float64)
    A = np.stack([X, np.ones_like(X)], axis=1)
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    s, t = float(coef[0]), float(coef[1])
    return s, t


def apply_calibration(c_raw, s, t):
    return s * c_raw + t


# -----------------------------
# Map generation
# -----------------------------
def make_iodine_maps_wi(roiconf, pair, alpha1, beta1, alpha2, beta2, s, t):
    """
    For each rep:
      - take map_slice (or middle of slice_range)
      - compute c_raw via 2x2 inversion
      - calibrate to mg/mL
      - save PNG
    """
    E1, E2 = pair
    s0, s1 = roiconf.get("slice_range", list(C.SLICE_RANGE))
    map_slice = roiconf.get("map_slice", int((int(s0) + int(s1)) // 2))

    out_dir = ensure_dir(C.OUT_DIR / "real_maps_wi" / f"{E1}_{E2}")
    reps = sorted({int(k.split("_")[1]) for k in C.SERIES.keys()})

    for rep in reps:
        f1 = Path(C.SERIES.get(f"{E1}_{rep}"))
        f2 = Path(C.SERIES.get(f"{E2}_{rep}"))
        if (not f1.exists()) or (not f2.exists()):
            continue

        dss1 = load_series_slices(f1)
        dss2 = load_series_slices(f2)
        idx = pick_slice_index(dss1, dss2, map_slice)

        hu1 = dcm_to_hu(dss1[idx])
        hu2 = dcm_to_hu(dss2[idx])

        _, c_raw, _ = solve_2x2(alpha1, beta1, alpha2, beta2, hu1, hu2)
        c_map = apply_calibration(c_raw, s, t)

        plt.figure()
        plt.imshow(c_map, interpolation="nearest")
        plt.colorbar(label="Iodine (mg/mL) [WI decomposition + calibration]")
        plt.title(f"Iodine map (WI) {E1}/{E2} rep={rep} slice={idx}")
        plt.tight_layout()
        plt.savefig(out_dir / f"iodine_map_wi_{E1}_{E2}_rep{rep}_slice{idx}.png", dpi=250)
        plt.close()


# -----------------------------
# Metrics
# -----------------------------
def summarize_metrics_from_rois(df, E1, E2, alpha1, beta1, alpha2, beta2, s, t, concs):
    """
    For each rep and each ROI concentration:
      - compute c_raw from HU means
      - calibrate -> c_pred
    Then summarize by (pair, conc_true): mean/std/bias/rmse/cv
    """
    d1 = df[df["kvp"] == E1][["rep"] + [f"c{c}" for c in concs]].copy()
    d2 = df[df["kvp"] == E2][["rep"] + [f"c{c}" for c in concs]].copy()
    m = d1.merge(d2, on="rep", suffixes=(f"_{E1}", f"_{E2}"))

    recs = []
    for _, row in m.iterrows():
        rep = int(row["rep"])
        for ctrue in concs:
            y1 = float(row[f"c{ctrue}_{E1}"])
            y2 = float(row[f"c{ctrue}_{E2}"])
            _, c_raw, _ = solve_2x2(alpha1, beta1, alpha2, beta2, y1, y2)
            c_pred = apply_calibration(c_raw, s, t)
            recs.append({"pair": f"{E1}/{E2}", "rep": rep, "conc_true": float(ctrue), "conc_pred": float(c_pred)})

    pred = pd.DataFrame(recs)
    g = pred.groupby(["pair", "conc_true"])
    rows = []
    for (pair, c), d in g:
        vals = d["conc_pred"].to_numpy(np.float64)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        bias = float(mean - c)
        rmse = float(np.sqrt(np.mean((vals - c) ** 2)))
        cv = float(std / (mean + 1e-12))
        rows.append({
            "pair": pair,
            "conc_true_mgml": float(c),
            "n_rep": int(len(vals)),
            "pred_mean": mean,
            "pred_std": std,
            "bias": bias,
            "rmse": rmse,
            "cv": cv,
        })

    return pred, pd.DataFrame(rows).sort_values(["pair", "conc_true_mgml"]).reset_index(drop=True)


# -----------------------------
# Main
# -----------------------------
def main():
    C.OUT_DIR.mkdir(exist_ok=True, parents=True)

    roiconf = json.loads(Path(C.ROIS_JSON).read_text(encoding="utf-8"))
    rois = roiconf["rois"]
    roi_concs = sorted({int(r["conc_mgml"]) for r in rois})

    # choose water ROI conc=0
    if 0 not in roi_concs:
        raise RuntimeError("rois.json must contain a water ROI with conc_mgml = 0")
    water_c = 0

    # choose reference iodine ROI as max concentration available
    ref_c = int(max(roi_concs))
    if ref_c == 0:
        raise RuntimeError("Need at least one iodine ROI with conc_mgml > 0")

    # calibration concs: use those present in rois.json (prefer config list)
    calib_concs = [int(c) for c in C.CALIB_CONCS if int(c) in roi_concs]
    if len(calib_concs) < 2:
        # fallback: use all nonzero concs
        calib_concs = [c for c in roi_concs if c > 0]
    calib_concs = sorted(calib_concs)

    # 1) ROI HU table
    df = compute_roi_hu_table(roiconf)
    df.to_csv(C.OUT_DIR / "real_roi_hu_table.csv", index=False)

    coeff_rows = []
    all_pred_rows = []
    all_summary_rows = []

    for (E1, E2) in C.ENERGY_PAIRS:
        # 2) estimate alpha/beta per energy
        alpha1, beta1 = estimate_alpha_beta(df, E1, water_c=water_c, ref_c=ref_c)
        alpha2, beta2 = estimate_alpha_beta(df, E2, water_c=water_c, ref_c=ref_c)

        # condition number of A = [[alpha1,beta1],[alpha2,beta2]]
        A = np.array([[alpha1, beta1],
                      [alpha2, beta2]], dtype=float)
        condA = float(np.linalg.cond(A))
        detA = float(np.linalg.det(A))

        # 3) fit final calibration on c_raw -> mg/mL
        s, t = fit_linear_calibration_from_rois(
            df, E1, E2, alpha1, beta1, alpha2, beta2, concs=calib_concs
        )

        coeff_rows.append({
            "pair": f"{E1}/{E2}",
            "water_c": water_c,
            "ref_c": ref_c,
            "alpha_E1": alpha1, "beta_E1": beta1,
            "alpha_E2": alpha2, "beta_E2": beta2,
            "cond_A": condA, "det_A": detA,
            "calib_concs": str(calib_concs),
            "s": s, "t": t
        })

        # 4) iodine maps (WI decomposition + calibration)
        make_iodine_maps_wi(roiconf, (E1, E2), alpha1, beta1, alpha2, beta2, s, t)

        # 5) ROI-level metrics
        pred_df, summ_df = summarize_metrics_from_rois(
            df, E1, E2, alpha1, beta1, alpha2, beta2, s, t, concs=roi_concs
        )
        all_pred_rows.append(pred_df)
        all_summary_rows.append(summ_df)

        # quick calibration scatter
        plt.figure()
        plt.scatter(pred_df["conc_true"], pred_df["conc_pred"])
        plt.xlabel("True iodine (mg/mL)")
        plt.ylabel("Predicted iodine (mg/mL)")
        plt.title(f"WI decomposition calibration check {E1}/{E2}")
        plt.tight_layout()
        plt.savefig(C.OUT_DIR / f"calib_check_wi_{E1}_{E2}.png", dpi=250)
        plt.close()

    coeffs = pd.DataFrame(coeff_rows)
    coeffs.to_csv(C.OUT_DIR / "real_wi_coeffs.csv", index=False)

    pred_all = pd.concat(all_pred_rows, ignore_index=True)
    pred_all.to_csv(C.OUT_DIR / "real_wi_pred_all.csv", index=False)

    summ_all = pd.concat(all_summary_rows, ignore_index=True)
    summ_all.to_csv(C.OUT_DIR / "real_wi_results_pairwise.csv", index=False)

    print("Saved:")
    print(" - outputs/real_wi_coeffs.csv")
    print(" - outputs/real_wi_pred_all.csv")
    print(" - outputs/real_wi_results_pairwise.csv")
    print(" - outputs/real_maps_wi/<pair>/*.png")
    print(" - outputs/calib_check_wi_*.png")


if __name__ == "__main__":
    main()
