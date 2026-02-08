import json
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from config import SERIES, ROIS_JSON, SLICE_RANGE, CALIB_CONCS, ALL_CONCS, ENERGY_PAIRS

s0, s1 = SLICE_RANGE
calib_concs = CALIB_CONCS
concs = ALL_CONCS
pairs = ENERGY_PAIRS


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
            if hasattr(ds, "InstanceNumber"):
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
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    roiconf = json.loads(ROIS_JSON.read_text(encoding="utf-8"))
    s0, s1 = roiconf["slice_range"]
    rois = roiconf["rois"]

    rows = []
    for sid, folder in SERIES.items():
        dss = load_series_slices(folder)

        # 取 slice 30..34（InstanceNumber）
        target = [ds for ds in dss if s0 <= int(ds.InstanceNumber) <= s1]
        if len(target) != (s1 - s0 + 1):
            raise RuntimeError(f"{sid}: expected {s1-s0+1} slices for {s0}-{s1}, got {len(target)}")

        # 逐 slice 计算 ROI mean，再对 5 层平均
        roi_means_per_slice = []
        for ds in target:
            hu = dcm_to_hu(ds)
            h, w = hu.shape
            vals = {}
            for r in rois:
                m = circular_mask(h, w, r["x"], r["y"], r["r_px"])
                vals[f"c{r['conc_mgml']}"] = float(np.mean(hu[m]))
            roi_means_per_slice.append(vals)

        # 5-slice average
        keys = roi_means_per_slice[0].keys()
        avg = {k: float(np.mean([d[k] for d in roi_means_per_slice])) for k in keys}

        # parse kvp from key
        kvp = int(sid.split("_")[0])
        rep = int(sid.split("_")[1])

        rows.append({"sid": sid, "kvp": kvp, "rep": rep, **avg})

    df = pd.DataFrame(rows).sort_values(["kvp", "rep"])
    df.to_csv(OUTDIR / "real_roi_hu_table.csv", index=False)
    print("Saved:", OUTDIR / "real_roi_hu_table.csv")

    # ====== calibration: for each energy pair fit c = a*HU_low + b*HU_high + d ======
    concs = [r["conc_mgml"] for r in rois]  # [0..400]
    conc_cols = [f"c{c}" for c in concs]

    # 主标定用 0..200（更线性）
    calib_concs = [0, 50, 100, 175, 200]
    calib_cols = [f"c{c}" for c in calib_concs]

    pairs = [(80,100), (100,120), (80,120)]
    results = []

    for (E1, E2) in pairs:
        # 用每个 rep 的 ROI 点一起拟合（等于把重复也当样本，提高稳定性）
        d1 = df[df["kvp"] == E1].reset_index(drop=True)
        d2 = df[df["kvp"] == E2].reset_index(drop=True)

        # 组装样本：每个 rep、每个浓度一个点
        X = []
        y = []
        for i in range(len(d1)):
            for ccol, c in zip(calib_cols, calib_concs):
                X.append([d1.loc[i, ccol], d2.loc[i, ccol], 1.0])
                y.append(c)
        X = np.asarray(X, np.float64)
        y = np.asarray(y, np.float64)

        # least squares
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b, d0 = coef.tolist()

        # 评价：在所有浓度上算预测
        all_pred = []
        all_true = []
        for i in range(len(d1)):
            for ccol, c in zip(conc_cols, concs):
                pred = a*d1.loc[i, ccol] + b*d2.loc[i, ccol] + d0
                all_pred.append(pred)
                all_true.append(c)
        all_pred = np.asarray(all_pred)
        all_true = np.asarray(all_true)

        rmse = float(np.sqrt(np.mean((all_pred - all_true) ** 2)))
        results.append({"pair": f"{E1}/{E2}", "a": a, "b": b, "d": d0, "rmse_all": rmse})

        # calibration plot
        plt.figure()
        plt.scatter(all_true, all_pred)
        plt.xlabel("True iodine (mg/mL)")
        plt.ylabel("Predicted iodine (mg/mL)")
        plt.title(f"Calibration check {E1}/{E2} (fit on 0-200)")
        plt.savefig(OUTDIR / f"calib_check_{E1}_{E2}.png", dpi=200, bbox_inches="tight")
        plt.close()

    pd.DataFrame(results).to_csv(OUTDIR / "calibration_coeffs.csv", index=False)
    print("Saved:", OUTDIR / "calibration_coeffs.csv")
    print("Saved calibration plots: outputs/calib_check_*.png")

if __name__ == "__main__":
    main()
