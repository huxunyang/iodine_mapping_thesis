import numpy as np
import pandas as pd

import config as C
from io_utils.nrrd_zip import read_nrrd_from_zip, list_members
from decomposition.img_domain import fit_effective_mu_2mat, decompose_WI_image
from evaluation.metrics import rmse, rel_rmse
from evaluation.tables import save_df
from viz.plots import plot_metric_vs_deltaE, scatter_x_y

def m(name: str) -> str:
    return C.SIM_BASE_DIR + name

def load_reconnoise_by_energy():
    pid = C.SIM_PHANTOM_ID
    members = list_members(C.SIM_ZIP_PATH, prefix=C.SIM_BASE_DIR)
    recon = {}
    for n in members:
        if f"ReconNoise_{pid}_E0" in n and n.endswith(".nrrd"):
            e = int(n.split("_E0")[-1].split(".")[0])  # E0080 -> 80
            arr, _ = read_nrrd_from_zip(C.SIM_ZIP_PATH, n)
            recon[e] = arr.astype(np.float32)
    return recon

def load_recon_gt_materials():
    pid = C.SIM_PHANTOM_ID
    W, _ = read_nrrd_from_zip(C.SIM_ZIP_PATH, m(f"ReconMuVolume_{pid}_Water.nrrd"))
    I, _ = read_nrrd_from_zip(C.SIM_ZIP_PATH, m(f"ReconMuVolume_{pid}_Iodine.nrrd"))
    B, _ = read_nrrd_from_zip(C.SIM_ZIP_PATH, m(f"ReconMuVolume_{pid}_Bone.nrrd"))
    T, _ = read_nrrd_from_zip(C.SIM_ZIP_PATH, m(f"ReconMuVolume_{pid}_Tissue.nrrd"))
    return W.astype(np.float32), I.astype(np.float32), B.astype(np.float32), T.astype(np.float32)

def bone_mask_from_gt(B_gt, frac=0.2):
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
    recon = load_reconnoise_by_energy()
    energies = sorted(recon.keys())
    print("Found recon energies:", energies)

    W_gt, I_gt, B_gt, T_gt = load_recon_gt_materials()
    bone_mask, bone_thr = bone_mask_from_gt(B_gt, frac=0.2)
    print(f"Bone mask voxels: {bone_mask.sum()} (thr={bone_thr:.6f})")

    rows = []
    for e1, e2 in C.SIM_PAIRS:
        if e1 not in recon or e2 not in recon:
            print(f"Skip pair {e1}/{e2} (missing recon energy)")
            continue

        Y1 = recon[e1]
        Y2 = recon[e2]

        # 1) 先拟合每个能量下的有效mu（2材料：W/I）
        muW1, muI1 = fit_effective_mu_2mat(W_gt, I_gt, Y1, sample=200000, seed=0)
        muW2, muI2 = fit_effective_mu_2mat(W_gt, I_gt, Y2, sample=200000, seed=0)

        # 2) 条件数（机制解释）
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
            "bone_B_thr": bone_thr,
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
