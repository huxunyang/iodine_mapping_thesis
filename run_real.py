import json
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt

import config as C  # 统一从 C 取配置


def dcm_to_hu(ds):
    px = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return px * slope + intercept


def load_series_slices(folder: Path):
    files = sorted([p for p in folder.rglob("*") if p.is_file()], key=lambda p: p.name)
    dss = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
            # 只保留真的 CT 图像切片
            if hasattr(ds, "InstanceNumber") and hasattr(ds, "PixelData"):
                dss.append(ds)
        except Exception:
            pass

    if not dss:
        raise RuntimeError(f"No DICOM slices found in {folder}")

    dss.sort(key=lambda ds: int(ds.InstanceNumber))
    return dss


def circular_mask(h, w, x, y, r):
    yy, xx = np.ogrid[:h, :w]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2


def main():
    # 如果你 realData 还是 zip，就打开这行
    # C.ensure_real_extracted()

    C.OUT_DIR.mkdir(exist_ok=True, parents=True)

    if not C.ROIS_JSON.exists():
        raise FileNotFoundError(f"Missing ROIS_JSON: {C.ROIS_JSON}")

    roiconf = json.loads(C.ROIS_JSON.read_text(encoding="utf-8"))
    s0, s1 = roiconf.get("slice_range", list(C.SLICE_RANGE))
    rois = roiconf["rois"]

    rows = []
    for sid, folder in C.SERIES.items():
        if not folder.exists():
            raise FileNotFoundError(f"{sid}: series folder not found: {folder}")

        dss = load_series_slices(folder)

        # 取 slice s0..s1（按 InstanceNumber）
        target = [ds for ds in dss if s0 <= int(ds.InstanceNumber) <= s1]
        expected = (s1 - s0 + 1)
        if len(target) != expected:
            raise RuntimeError(f"{sid}: expected {expected} slices for {s0}-{s1}, got {len(target)}")

        # 逐 slice 计算 ROI mean，再对多层平均
        roi_means_per_slice = []
        for ds in target:
            hu = dcm_to_hu(ds)
            h, w = hu.shape
            vals = {}
            for r in rois:
                m = circular_mask(h, w, r["x"], r["y"], r["r_px"])
                vals[f"c{r['conc_mgml']}"] = float(np.mean(hu[m]))
            roi_means_per_slice.append(vals)

        keys = roi_means_per_slice[0].keys()
        avg = {k: float(np.mean([d[k] for d in roi_means_per_slice])) for k in keys}

        kvp = int(sid.split("_")[0])
        rep = int(sid.split("_")[1])
        rows.append({"sid": sid, "kvp": kvp, "rep": rep, **avg})

    df = pd.DataFrame(rows).sort_values(["kvp", "rep"]).reset_index(drop=True)
    df.to_csv(C.OUT_DIR / "real_roi_hu_table.csv", index=False)
    print("Saved:", C.OUT_DIR / "real_roi_hu_table.csv")

    # ====== calibration: for each energy pair fit c = a*HU_low + b*HU_high + d ======
    roi_concs = [r["conc_mgml"] for r in rois]
    conc_cols = [f"c{c}" for c in roi_concs]

    calib_concs = C.CALIB_CONCS
    calib_cols = [f"c{c}" for c in calib_concs]

    results = []
    for (E1, E2) in C.ENERGY_PAIRS:
        d1 = df[df["kvp"] == E1].reset_index(drop=True)
        d2 = df[df["kvp"] == E2].reset_index(drop=True)

        if len(d1) != len(d2) or len(d1) == 0:
            raise RuntimeError(f"Pair {E1}/{E2}: kvp groups mismatch or empty (E1={len(d1)}, E2={len(d2)})")

        X, y = [], []
        for i in range(len(d1)):
            for ccol, c in zip(calib_cols, calib_concs):
                X.append([d1.loc[i, ccol], d2.loc[i, ccol], 1.0])
                y.append(c)

        X = np.asarray(X, np.float64)
        y = np.asarray(y, np.float64)

        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b, d0 = coef.tolist()

        all_pred, all_true = [], []
        for i in range(len(d1)):
            for ccol, c in zip(conc_cols, roi_concs):
                pred = a * d1.loc[i, ccol] + b * d2.loc[i, ccol] + d0
                all_pred.append(pred)
                all_true.append(c)

        all_pred = np.asarray(all_pred)
        all_true = np.asarray(all_true)
        rmse_all = float(np.sqrt(np.mean((all_pred - all_true) ** 2)))

        results.append({"pair": f"{E1}/{E2}", "a": a, "b": b, "d": d0, "rmse_all": rmse_all})

        plt.figure()
        plt.scatter(all_true, all_pred)
        plt.xlabel("True iodine (mg/mL)")
        plt.ylabel("Predicted iodine (mg/mL)")
        plt.title(f"Calibration check {E1}/{E2} (fit on {calib_concs[0]}-{calib_concs[-1]})")
        plt.tight_layout()
        plt.savefig(C.OUT_DIR / f"calib_check_{E1}_{E2}.png", dpi=300)
        plt.close()

    pd.DataFrame(results).to_csv(C.OUT_DIR / "calibration_coeffs.csv", index=False)
    print("Saved:", C.OUT_DIR / "calibration_coeffs.csv")
    print("Saved calibration plots:", C.OUT_DIR / "calib_check_*.png")


if __name__ == "__main__":
    main()
