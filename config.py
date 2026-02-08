from pathlib import Path
import zipfile

# config.py 所在：workplace/iodine_mapping_thesis/config.py
PROJECT_ROOT = Path(__file__).resolve().parent   # 或 parent.parent 取决于你config放哪层
WORKPLACE_ROOT = PROJECT_ROOT.parent             # => workplace/
DATA_ROOT = WORKPLACE_ROOT / "thesisData/Data"              # => thesisData/Data

OUT_DIR = PROJECT_ROOT / "outputs"               # 输出放项目里最方便


# ----------------
# Simulation data
# ----------------
SIM_DATA_ROOT = DATA_ROOT / "simData"  # 解压目标根目录

SIM_PHANTOM_DIRNAME = "phantom2"   # zip里目录名: simData/phantom2/
SIM_FILE_PID = "Phantom2"          # 文件名中的 Phantom ID: ProjMuVolume_Phantom2_...
SIM_PHANTOM_DIR = SIM_DATA_ROOT / SIM_PHANTOM_DIRNAME

ENERGIES = [80, 100, 120]
SIM_PAIRS = [(80, 100), (100, 120), (80, 120)]

# 骨泄漏阈值（projection-domain bone_leakage_metric 用）
BONE_RAY_THRESH = 0.2
BONE_VOL_FRAC = 0.2
# --------------
# Real data
# --------------
REAL_DATA_ROOT = DATA_ROOT / "realData"  # 解压目标根目录

# real series（run_real.py 用）
# sid 统一定义为: "{kvp}_{rep}"，例如 "80_1"
SERIES = {
    f"{kvp}_{rep}": (REAL_DATA_ROOT / f"{kvp}kvp" / str(rep))
    for kvp in [80, 100, 120]
    for rep in [1, 2, 3]
}

# 真实数据 ROI 配置（你需要自己准备一个 json 文件）
# 建议放在: outputs/rois.json
ROIS_JSON = OUT_DIR / "rois.json"

# 默认 slice 范围（也可以写进 rois.json 覆盖）
SLICE_RANGE = (30, 34)

# 标定用浓度、全浓度（按你phantom实际来）
CALIB_CONCS = [0, 50, 100, 175, 200]
ALL_CONCS   = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400]

ENERGY_PAIRS = [(80, 100), (100, 120), (80, 120)]
