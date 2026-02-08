import numpy as np
import pandas as pd

import config as C
from io_utils.nrrd_dir import read_nrrd_from_dir

from decomposition.img_domain import fit_effective_mu_2mat, decompose_WI_image
from evaluation.metrics import rmse, rel_rmse
from evaluation.tables import save_df
from viz.plots import plot_metric_vs_deltaE, scatter_x_y


def _energy_tag(e: int) -> str:
    """
    你的文件名能量标签是：
      80  -> E080  (3位)
      100 -> E0100 (4位)
      120 -> E0120 (4位)
    """
    return f"{e:03d}" if e < 100 else f"{e:04d}"


def _fname_noise(kind: str, energy: int) -> str:
    # kind: "ReconNoise" 等
    return f"{kind}_{C.SIM_FILE_PID}_E{_energy_tag(energy)}.nrrd"


def _fname_mu(kind: str, material: str) -> str:
    """
    GT 材料体数据文件名（解压目录下）：
      ReconMuVolume_Phantom2_Water.nrrd
      ReconMuVolume_Phantom2_Iodine.nrrd
      ReconMuVolume_Phantom2_Bone.nrrd
      ReconMuVolume_Phantom2_Tissue.nrrd
    """
    return f"{kind}_{C.SIM_FILE_PID}_{material}.nrrd"


def load_reconnoise_by_energy():
    """
    在文件夹里扫描：
      ReconNoise_Phantom2_E080.nrrd
      ReconNoise_Phantom2_E0100.nrrd
      ReconNoise_Phantom2_E0120.nrrd
    """
    base = C.SIM_PHANTOM_DIR
    pid = C.SIM_FILE_PID

    recon = {}
    pattern = f"ReconNoise_{pid}_E*.nrrd"
    files = sorted(base.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matched {pattern} under {base}. "
            f"Check SIM_PHANTOM_DIR and SIM_FILE_PID."
        )

    for fp in files:
        name = fp.name
        # 从 "ReconNoise_Phantom2_E080.nrrd" 或 "ReconNoise_Phantom2_E0100.nrrd" 提取能量
        try:
            e_str = name.split("_E")[-1].split(".")[0]  # "080"/"0100"/"0120"
            e = int(e_str)  # int("080")->80, int("0100")->100
        except Exception as ex:
            raise ValueError(f"Cannot parse energy from filename: {name}") from ex

        arr, _ = read_nrrd_from_dir(base, name)
        recon[e] = arr.astype(np.float32)

    return recon


def load_recon_gt_materials():
    base = C.SIM_PHANTOM_DIR

    W, _ = read_nrrd_from_dir(base, _fname_mu("ReconMuVolume", "Water"))
    I, _ = read_nrrd_from_dir(base, _fname_mu("ReconMuVolume", "Iodine"))
    B, _ = read_nrrd_from_dir(base, _fname_mu("ReconMuVolume", "Bone"))
    T, _ = read_nrrd_from_dir(base, _fname_mu("ReconMuVolume", "Tissue"))

    return W.astype(np.float32), I.astype(np.float32), B.astype(np.float32), T.astype(np.float32)


def bone_mask_from_gt(B_gt, frac=C.BONE_VOL_FRAC):
    """
    用骨GT生成骨ROI：mask = B_gt > frac * max(B_gt)
    frac=0.2 或 0.3 通常更稳健
    """
    mx = float(B_gt.max())
    if mx <= 0:
        return np.zeros_like(B_gt, dtype=bool), 0.0
    thr = frac * mx
    mask = B_gt > thr
    return mask, thr


def main():
    # 输出目录
    C.OUT_DIR.mkdir(exist_ok=True, parents=True)

    # 读重建噪声图像
    recon = load_reconnoise_by_energy()
    energies = sorted(recon.keys())
    print("SIM_PHANTOM_DIR =", C.SIM_PHANTOM_DIR)
    print("SIM_FILE_PID   =", C.SIM_FILE_PID)
    print("Found recon energies:", energies)

    # 读 GT 材料图像
    W_gt, I_gt, B_gt, T_gt = load_recon_gt_materials()

    # 骨区 mask
    bone_mask, bone_thr = bone_mask_from_gt(B_gt, frac=0.2)
    print(f"Bone mask voxels: {int(bone_mask.sum())} (thr={bone_thr:.6f})")

    rows = []
    for e1, e2 in C.SIM_PAIRS:
        if e1 not in recon or e2 not in recon:
            print(f"Skip pair {e1}/{e2} (missing recon energy)")
            continue

        Y1 = recon[e1]
        Y2 = recon[e2]

        # 1) 拟合每个能量下的有效mu（2材料：W/I）
        muW1, muI1 = fit_effective_mu_2mat(W_gt, I_gt, Y1, sample=200000, seed=0)
        muW2, muI2 = fit_effective_mu_2mat(W_gt, I_gt, Y2, sample=200000, seed=0)

        # 2) 条件数/行列式（机制解释）
        A = np.array([[muW1, muI1],
                      [muW2, muI2]], dtype=float)
        condA = float(np.linalg.cond(A))
        detA  = float(np.linalg.det(A))

        # 3) 图像域分解
        _, I_est = decompose_WI_image(Y1, Y2, muW1, muI1, muW2, muI2)

        # 4) 误差评估（对 GT）
        iodine_rmse = rmse(I_est, I_gt)
        iodine_rel  = rel_rmse(I_est, I_gt)

        # 5) 骨区假碘（图像域）
        bone_false_abs_mean = float(np.mean(np.abs(I_est[bone_mask]))) if bone_mask.any() else np.nan
        bone_false_mean     = float(np.mean(I_est[bone_mask])) if bone_mask.any() else np.nan

        rows.append({
            "pair": f"{e1}/{e2}",
            "deltaE": abs(e2 - e1),
            "cond_A": condA,
            "det_A": detA,
            "iodine_RMSE_vs_GT": iodine_rmse,
            "iodine_relRMSE_vs_GT": iodine_rel,
            "bone_falseI_mean": bone_false_mean,
            "bone_falseI_abs_mean": bone_false_abs_mean,
            "bone_voxels": int(bone_mask.sum()),
            "bone_B_thr": float(bone_thr),
        })

    df = pd.DataFrame(rows).sort_values("deltaE")
    out_csv = C.OUT_DIR / "sim_image_domain_results.csv"
    save_df(df, out_csv)

    print("\nSaved:", out_csv)
    print(df.to_string(index=False))

    # 图：relRMSE vs ΔE
    plot_metric_vs_deltaE(
        df, "iodine_relRMSE_vs_GT",
        C.OUT_DIR / "sim_img_iodine_relRMSE_vs_deltaE.png",
        title="Sim Image-domain: Iodine relRMSE vs ΔE"
    )

    # 图：骨假碘 vs ΔE
    plot_metric_vs_deltaE(
        df, "bone_falseI_abs_mean",
        C.OUT_DIR / "sim_img_bone_falseI_abs_mean_vs_deltaE.png",
        title="Sim Image-domain: Bone false iodine vs ΔE"
    )

    # 图：机制（relRMSE vs condA）
    scatter_x_y(
        df, "cond_A", "iodine_relRMSE_vs_GT",
        C.OUT_DIR / "sim_img_relRMSE_vs_condA.png",
        title="Sim Image-domain: relRMSE vs cond(A)",
        annotate_col="pair"
    )

    print("Saved image-domain plots to outputs/")


if __name__ == "__main__":
    main()
