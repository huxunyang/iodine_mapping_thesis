import pandas as pd
import numpy as np

import config.sim as C
from io_utils.nrrd_dir import read_nrrd_from_dir

from decomposition.proj_domain import fit_mu_coeffs, decompose_WI
from evaluation.metrics import rmse, rel_rmse, bone_leakage_metric
from evaluation.tables import save_df
from viz.plots import plot_metric_vs_deltaE, scatter_x_y


def _fname(pid: str, material: str) -> str:
    # 适配你的文件命名：ProjMuVolume_{Phantom2}_Water.nrrd
    return f"ProjMuVolume_{pid}_{material}.nrrd"


def load_material_projs():
    pid = C.SIM_PHANTOM_ID
    base = C.SIM_PHANTOM_DIR

    LW, _ = read_nrrd_from_dir(base, _fname(pid, "Water"))
    LI, _ = read_nrrd_from_dir(base, _fname(pid, "Iodine"))
    LB, _ = read_nrrd_from_dir(base, _fname(pid, "Bone"))
    LT, _ = read_nrrd_from_dir(base, _fname(pid, "Tissue"))
    return LW, LI, LB, LT


def load_projnoise_by_energy():
    """
    在文件夹里扫描：
      ProjNoise_{pid}_E0080.nrrd
      ProjNoise_{pid}_E0100.nrrd
      ProjNoise_{pid}_E0120.nrrd
    """
    pid = C.SIM_PHANTOM_ID
    base = C.SIM_PHANTOM_DIR

    proj = {}
    pattern = f"ProjNoise_{pid}_E0*.nrrd"
    files = sorted(base.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matched {pattern} under {base}. "
            f"Check SIM_PHANTOM_DIR and SIM_PHANTOM_ID."
        )

    for fp in files:
        name = fp.name  # 文件名
        # 从 "ProjNoise_Phantom2_E0080.nrrd" 提取 80
        try:
            e_str = name.split("_E0")[-1].split(".")[0]  # "080" or "0800" depending
            e = int(e_str)  # "0080" -> 80
        except Exception as ex:
            raise ValueError(f"Cannot parse energy from filename: {name}") from ex

        arr, _ = read_nrrd_from_dir(base, name)
        proj[e] = arr

    return proj


def main():
    # 0) 检查路径配置
    print("SIM_PHANTOM_DIR =", C.SIM_PHANTOM_DIR)
    print("SIM_PHANTOM_ID  =", C.SIM_PHANTOM_ID)

    # 1) 读材料投影真值 & 噪声投影
    LW, LI_gt, LB, LT = load_material_projs()
    proj = load_projnoise_by_energy()
    energies = sorted(proj.keys())
    print("Found energies:", energies)

    # 2) 每能量拟合有效系数 muW muI muB muT
    mu = {}
    for e in energies:
        coef = fit_mu_coeffs(LW, LI_gt, LB, LT, proj[e], sample=200000, seed=0)
        mu[e] = coef
        print(
            f"E={e:3d}: muW={coef[0]:.6f}, muI={coef[1]:.6f}, muB={coef[2]:.6f}, muT={coef[3]:.6f}"
        )

    # 3) 能量对分解 + 指标
    rows = []
    for e1, e2 in C.SIM_PAIRS:
        if e1 not in proj or e2 not in proj:
            print(f"Skip pair {e1}/{e2} (missing energy)")
            continue

        muW1, muI1 = mu[e1][0], mu[e1][1]
        muW2, muI2 = mu[e2][0], mu[e2][1]

        _, LI_est = decompose_WI(proj[e1], proj[e2], muW1, muI1, muW2, muI2)

        A = np.array([[muW1, muI1],
                      [muW2, muI2]], dtype=float)
        condA = float(np.linalg.cond(A))
        detA = float(np.linalg.det(A))

        leak = bone_leakage_metric(LI_est, LB, thresh=C.BONE_RAY_THRESH)

        rows.append({
            "pair": f"{e1}/{e2}",
            "deltaE": abs(e2 - e1),
            "cond_A": condA,
            "det_A": detA,
            "iodine_RMSE_vs_GT": rmse(LI_est, LI_gt),
            "iodine_relRMSE_vs_GT": rel_rmse(LI_est, LI_gt),
            "bone_falseI_mean": leak["falseI_mean"],
            "bone_falseI_abs_mean": leak["falseI_abs_mean"],
            "bone_mask_count": leak["bone_mask_count"],
            "bone_LB_thresh": leak["lb_thresh"],
        })

    df = pd.DataFrame(rows).sort_values("deltaE")
    out_csv = C.OUT_DIR / "sim_projection_domain_results.csv"
    save_df(df, out_csv)

    # 4) 画图
    plot_metric_vs_deltaE(
        df, "iodine_relRMSE_vs_GT",
        C.OUT_DIR / "sim_iodine_relRMSE_vs_deltaE.png",
        title="Simulation: Iodine relRMSE vs ΔE"
    )
    plot_metric_vs_deltaE(
        df, "bone_falseI_abs_mean",
        C.OUT_DIR / "sim_bone_falseI_abs_mean_vs_deltaE.png",
        title="Simulation: Bone false iodine vs ΔE"
    )
    scatter_x_y(
        df, "cond_A", "iodine_relRMSE_vs_GT",
        C.OUT_DIR / "sim_relRMSE_vs_condA.png",
        title="Simulation: relRMSE vs cond(A)",
        annotate_col="pair"
    )

    print("Saved plots to outputs/")
    print("\nSaved:", out_csv)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
