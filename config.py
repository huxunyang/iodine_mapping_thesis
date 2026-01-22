from pathlib import Path

OUT_DIR = Path('outputs')
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ===== 仿真数据（请改成你本机 zip 路径）=====
SIM_ZIP_PATH = Path(r'D:\data\phantom 0.zip')   # <-- 改这里
SIM_BASE_DIR = 'phantom 0/'                     # zip内部目录
SIM_PHANTOM_ID = 'Phantom0'                     # Phantom0 或 Phantom1

SIM_PAIRS = [(80, 100), (100, 120), (80, 120)]
BONE_RAY_THRESH = None

# ===== 真实数据（先占位，后续填）=====
REAL_SERIES_DIRS = {
    80: Path(r'D:\data\real\80kvp'),
    100: Path(r'D:\data\real\100kvp'),
    120: Path(r'D:\data\real\120kvp'),
}
REAL_PAIRS = [(80, 100), (100, 120), (80, 120)]
