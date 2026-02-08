from pathlib import Path
from .common import ENERGY_PAIRS

# 真实数据根目录
DATA_ROOT = Path(r"E:\Data\Xunyang_HU\OVGU\Master_Thesis\thesisData\Data\realData")

# ROI 与 slice
ROIS_JSON = Path("outputs/rois.json")
SLICE_RANGE = (30, 34)

# 9 次扫描（注意：Windows 路径用 / 拼接，避免 \1 被当成转义）
REAL_SERIES = {
    "80_1":  DATA_ROOT / "80kvp" / "1",
    "80_2":  DATA_ROOT / "80kvp" / "2",
    "80_3":  DATA_ROOT / "80kvp" / "3",

    "100_1": DATA_ROOT / "100kvp" / "1",
    "100_2": DATA_ROOT / "100kvp" / "2",
    "100_3": DATA_ROOT / "100kvp" / "3",

    "120_1": DATA_ROOT / "120kvp" / "1",
    "120_2": DATA_ROOT / "120kvp" / "2",
    "120_3": DATA_ROOT / "120kvp" / "3",
}

REAL_PAIRS = ENERGY_PAIRS
