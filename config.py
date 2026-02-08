from pathlib import Path

OUT_DIR = Path('outputs')
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ===== 仿真数据（请改成你本机 zip 路径）=====
SIM_ZIP_PATH = Path(r'E:\Data\Xunyang_HU\OVGU\Master_Thesis\thesisData\Data.zip')   # <-- 改这里
SIM_BASE_DIR = 'Data/simData/phantom2/'                     # zip内部目录
SIM_PHANTOM_ID = 'Phantom2'                     # Phantom0 或 Phantom1

SIM_PAIRS = [(80, 100), (100, 120), (80, 120)]
BONE_RAY_THRESH = None

# ===== 真实数据（先占位，后续填）=====
DATA_ROOT = Path(r"E:\Data\Xunyang_HU\OVGU\Master_Thesis\thesisData\Data\realData")

# ===== ROI 配置 =====
ROIS_JSON = Path("outputs/rois.json")
SLICE_RANGE = (30, 34)

# ===== 真实 CT series（9 次扫描）=====
SERIES = {
    "80_1":  DATA_ROOT / "80kvp\1",
    "80_2":  DATA_ROOT / "80kvp\2",
    "80_3":  DATA_ROOT / "80kvp\3",

    "100_1": DATA_ROOT / "100kvp\1",
    "100_2": DATA_ROOT / "100kvp\2",
    "100_3": DATA_ROOT / "100kvp\3",

    "120_1": DATA_ROOT / "120kvp\1",
    "120_2": DATA_ROOT / "120kvp\2",
    "120_3": DATA_ROOT / "120kvp\3",
}

# ===== 标定设置 =====
CALIB_CONCS = [0, 50, 100, 175, 200]     # 主标定
ALL_CONCS   = [0, 50, 100, 175, 200, 300, 350, 400]

ENERGY_PAIRS = [(80,100), (100,120), (80,120)]

