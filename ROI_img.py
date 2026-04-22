import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import pydicom

from config import REAL_DATA_ROOT


# =========================
# 1. 路径设置（按你的实际路径修改）
# =========================
dicom_dir = REAL_DATA_ROOT / "120kvp" / "1"      # 120 kVp, repetition 1
rois_json = Path(r"outputs/rois.json")       # ROI 文件
out_png = Path(r"outputs/roi_overlay_120kvp_rep1_slice32_legend_inside.png")

target_slice_index = 32   # 第32张 slice


# =========================
# 2. DICOM 转 HU
# =========================
def dcm_to_hu(ds):
    px = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return px * slope + intercept


# =========================
# 3. 读取并排序 DICOM 序列
# =========================
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


# =========================
# 4. 读取 ROI 配置
# =========================
with open(rois_json, "r", encoding="utf-8") as f:
    roi_conf = json.load(f)

rois = roi_conf["rois"]

# 按浓度排序，图例更清楚
rois = sorted(rois, key=lambda r: r["conc_mgml"])


# =========================
# 5. 读取目标 slice
# =========================
dss = load_series_slices(dicom_dir)

if target_slice_index < 0 or target_slice_index >= len(dss):
    raise IndexError(
        f"target_slice_index={target_slice_index} out of range, "
        f"but only {len(dss)} slices found."
    )

ds = dss[target_slice_index]
img_hu = dcm_to_hu(ds)


# =========================
# 6. 绘图：不同颜色 ROI + 图内右上角图例
# =========================
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(img_hu, cmap="gray")
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("HU")

# 颜色表
cmap = plt.get_cmap("tab10")
colors = [cmap(i % 10) for i in range(len(rois))]

legend_handles = []

for i, roi in enumerate(rois):
    x = float(roi["x"])          # column
    y = float(roi["y"])          # row
    r = float(roi["r_px"])
    roi_id = str(roi["id"])
    conc = roi["conc_mgml"]
    color = colors[i]

    # 画 ROI 圆
    circ = Circle((x, y), r, fill=False, linewidth=2.0, color=color)
    ax.add_patch(circ)

    # 图例项
    handle = Line2D(
        [0], [0],
        color=color,
        lw=2.5,
        label=f"{roi_id}: {conc} mg/mL"
    )
    legend_handles.append(handle)

ax.set_title("120 kVp, repetition 1, slice 32 with ROI overlay")
ax.set_xlabel("x (pixel)")
ax.set_ylabel("y (pixel)")
ax.set_xlim(0, img_hu.shape[1])
ax.set_ylim(img_hu.shape[0], 0)

# 图例放在原图右上角（图内）
ax.legend(
    handles=legend_handles,
    loc="upper right",
    frameon=True,
    facecolor="white",
    framealpha=0.85,
    title="ROIs",
    fontsize=8,
    title_fontsize=9
)

plt.tight_layout()
out_png.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved to: {out_png}")