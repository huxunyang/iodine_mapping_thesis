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
    """Load CT slices from a folder. Sort robustly by ImagePositionPatient or InstanceNumber."""
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


def pick_slices(dss, s0, s1):
    """Pick slice range by index in sorted list (safer than relying on InstanceNumber)."""
    n = len(dss)
    s0 = max(0, int(s0))
    s1 = min(n - 1, int(s1))
    if s0 > s1:
        raise ValueError(f"Invalid slice range: {s0}-{s1} for n={n}")
    return dss[s0 : s1 + 1]


def circular_mask(h, w, x, y, r):
    yy, xx = np.ogrid[:h, :w]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2


def ensure_dir(p: Path):
    p.mkdir(exist_ok=True, parents=True)
    return p


# -----------------------------
# Core pipeline
# -----------------------------
def compute_roi_hu_table(roiconf):
    """
    Return df with columns:
      sid, kvp, rep, c0, c50, ...
    Each cXXX is mean HU across slice_range (averaged over slices).
    """
    s0, s1 = roiconf.get("slice_range", list(C.SLICE_RANGE))
    rois = roiconf["rois"]

    rows = []
    for sid, folder in C.SERIES.items():
        if (folder is None) or (not Path(folder).exists()):
            raise FileNotFoundError(f"{sid}: series folder not found: {folder}")

        dss = load_series_slices(Path(folder))
        target = pick_slices(dss, s0, s1)

        per_slice = []
        for ds in target:
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

    df = pd.DataFrame(rows).sort_values(["kvp", "rep"]).reset_index(drop=True)
    return df


def fit_pair_calibration(df, E1, E2, calib_concs):
    """
    Fit: C = a*HU_E1 + b*HU_E2 + d
    IMPORTANT: align by rep to avoid mismatch.
    """
    d1 = df[df["kvp"] == E1][["rep"] + [f"c{c}" for c in calib_concs]].copy()
    d2 = df[df["kvp"] == E2][["rep"] + [f"c{c}" for c in calib_concs]].copy()

    m = d1.merge(d2, on="rep", suffixes=(f"_{E1}", f"_{E2}"))
    if len(m) == 0:
        raise RuntimeError(f"Pair {E1}/{E2}: merged table empty. Check reps and folders.")

    X, y = [], []
    for _, row in m.iterrows():
        for c in calib_concs:
            X.append([row[f"c{c}_{E1}"], row[f"c{c}_{E2}"], 1.0])
            y.append(float(c))

    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, d0 = coef.tolist()

    # condition number for the (a,b) mapping is not meaningful (not 2x2 inversion),
    # but we still output the regression coefficients for variation analysis.
    return a, b, d0


def predict_concs_from_rois(df, E1, E2, a, b, d0, all_concs):
    """Return per-rep predictions for each conc using ROI HU table."""
    d1 = df[df["kvp"] == E1][["rep"] + [f"c{c}" for c in all_concs]].copy()
    d2 = df[df["kvp"] == E2][["rep"] + [f"c{c}" for c in all_concs]].copy()
    m = d1.merge(d2, on="rep", suffixes=(f"_{E1}", f"_{E2}"))

    out = []
    for _, row in m.iterrows():
        rep = int(row["rep"])
        for c in all_concs:
            hu1 = float(row[f"c{c}_{E1}"])
            hu2 = float(row[f"c{c}_{E2}"])
            pred = a * hu1 + b * hu2 + d0
            out.append({"pair": f"{E1}/{E2}", "rep": rep, "conc_true": float(c), "conc_pred": float(pred)})
    return pd.DataFrame(out)


def make_iodine_map_for_pair(roiconf, E1, E2, a, b, d0):
    """
    Generate iodine map images for each rep:
      C_map = a*HU_E1 + b*HU_E2 + d0
    Save a PNG per rep for a single slice (map_slice) or middle of slice_range.
    """
    s0, s1 = roiconf.get("slice_range", list(C.SLICE_RANGE))
    map_slice = roiconf.get("map_slice", None)
    if map_slice is None:
        map_slice = int((s0 + s1) // 2)

    out_dir = ensure_dir(C.OUT_DIR / "real_maps" / f"{E1}_{E2}")

    for rep in sorted({int(k.split("_")[1]) for k in C.SERIES.keys()}):
        f1 = C.SERIES.get(f"{E1}_{rep}")
        f2 = C.SERIES.get(f"{E2}_{rep}")
        if f1 is None or f2 is None:
            continue

        dss1 = load_series_slices(Path(f1))
        dss2 = load_series_slices(Path(f2))

        # use the same slice index in sorted order
        idx = int(map_slice)
        idx = max(0, min(idx, min(len(dss1), len(dss2)) - 1))

        hu1 = dcm_to_hu(dss1[idx])
        hu2 = dcm_to_hu(dss2[idx])
        c_map = a * hu1 + b * hu2 + d0

        # save png
        plt.figure()
        plt.imshow(c_map, interpolation="nearest")
        plt.colorbar(label="Iodine (mg/mL)  [baseline regression]")
        plt.title(f"Iodine map (baseline)  {E1}/{E2}  rep={rep}  slice={idx}")
        plt.tight_layout()
        plt.savefig(out_dir / f"iodine_map_{E1}_{E2}_rep{rep}_slice{idx}.png", dpi=250)
        plt.close()


def summarize_metrics(pred_df):
    """
    pred_df columns: pair, rep, conc_true, conc_pred
    Output per-pair, per-conc summary: mean, std, bias, rmse, cv
    """
    g = pred_df.groupby(["pair", "conc_true"])
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
    return pd.DataFrame(rows).sort_values(["pair", "conc_true_mgml"]).reset_index(drop=True)


def main():
    C.OUT_DIR.mkdir(exist_ok=True, parents=True)

    if not C.ROIS_JSON.exists():
        raise FileNotFoundError(f"Missing ROIS_JSON: {C.ROIS_JSON}")

    roiconf = json.loads(C.ROIS_JSON.read_text(encoding="utf-8"))
    if "rois" not in roiconf:
        raise ValueError("rois.json must contain key: 'rois'")

    # 1) ROI HU table
    df = compute_roi_hu_table(roiconf)
    df_path = C.OUT_DIR / "real_roi_hu_table.csv"
    df.to_csv(df_path, index=False)
    print("Saved:", df_path)

    # all concs from rois.json (preferred)
    roi_concs = sorted({int(r["conc_mgml"]) for r in roiconf["rois"]})
    all_concs = [c for c in roi_concs]  # use what's present in rois.json
    # calibration concs from config, but keep only those existing in rois.json
    calib_concs = [c for c in C.CALIB_CONCS if int(c) in roi_concs]
    if len(calib_concs) < 2:
        raise RuntimeError(
            f"Not enough CALIB_CONCS present in rois.json. "
            f"CALIB_CONCS={C.CALIB_CONCS}, rois.json concs={roi_concs}"
        )

    # 2) fit per-pair calibration + save coeffs + plots
    coeff_rows = []
    pred_all = []

    for (E1, E2) in C.ENERGY_PAIRS:
        a, b, d0 = fit_pair_calibration(df, E1, E2, calib_concs)
        coeff_rows.append({"pair": f"{E1}/{E2}", "a": a, "b": b, "d": d0, "calib_concs": str(calib_concs)})

        # calibration check scatter using ALL concs available in rois.json
        pred_df = predict_concs_from_rois(df, E1, E2, a, b, d0, all_concs)
        pred_all.append(pred_df)

        plt.figure()
        plt.scatter(pred_df["conc_true"], pred_df["conc_pred"])
        plt.xlabel("True iodine (mg/mL)")
        plt.ylabel("Predicted iodine (mg/mL)")
        plt.title(f"Calibration check (baseline) {E1}/{E2}")
        plt.tight_layout()
        plt.savefig(C.OUT_DIR / f"calib_check_{E1}_{E2}.png", dpi=250)
        plt.close()

        # 3) iodine map images (baseline)
        make_iodine_map_for_pair(roiconf, E1, E2, a, b, d0)

    coeffs = pd.DataFrame(coeff_rows)
    coeff_path = C.OUT_DIR / "calibration_coeffs.csv"
    coeffs.to_csv(coeff_path, index=False)
    print("Saved:", coeff_path)

    pred_all = pd.concat(pred_all, ignore_index=True)
    pred_path = C.OUT_DIR / "real_pred_all.csv"
    pred_all.to_csv(pred_path, index=False)
    print("Saved:", pred_path)

    # 4) summary metrics table (per pair, per conc)
    summ = summarize_metrics(pred_all)
    summ_path = C.OUT_DIR / "real_results_pairwise.csv"
    summ.to_csv(summ_path, index=False)
    print("Saved:", summ_path)

    print("Saved iodine maps under:", (C.OUT_DIR / "real_maps").resolve())


if __name__ == "__main__":
    main()
