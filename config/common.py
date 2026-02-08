from pathlib import Path

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# 能量对（用于比较谱变化）
ENERGY_PAIRS = [(80, 100), (100, 120), (80, 120)]

# 浓度（mg/mL）
ALL_CONCS = [0, 50, 100, 175, 200, 300, 350, 400]
CALIB_CONCS = [0, 50, 100, 175, 200]  # 主标定（更线性、更稳）
