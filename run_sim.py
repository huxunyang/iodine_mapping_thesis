import pandas as pd

import config as C
from io_utils.nrrd_zip import read_nrrd_from_zip, list_members
from decomposition.proj_domain import fit_mu_coeffs, decompose_WI
from evaluation.metrics import rmse, rel_rmse, bone_leakage_metric
from evaluation.tables import save_df

def m(name: str) -> str:
    return C.SIM_BASE_DIR + name

def load_material_projs():
    pid = C.SIM_PHANTOM_ID
    LW, _ = read_nrrd_from_zip(C.SIM_ZIP_PATH, m(f'ProjMuVolume_{pid}_Water.nrrd'))
    LI, _ = read_nrrd_from_zip(C.SIM_ZIP_PATH, m(f'ProjMuVolume_{pid}_Iodine.nrrd'))
    LB, _ = read_nrrd_from_zip(C.SIM_ZIP_PATH, m(f'ProjMuVolume_{pid}_Bone.nrrd'))
    LT, _ = read_nrrd_from_zip(C.SIM_ZIP_PATH, m(f'ProjMuVolume_{pid}_Tissue.nrrd'))
    return LW, LI, LB, LT

def load_projnoise_by_energy():
    pid = C.SIM_PHANTOM_ID
    members = list_members(C.SIM_ZIP_PATH, prefix=C.SIM_BASE_DIR)
    proj = {}
    for n in members:
        if f'ProjNoise_{pid}_E0' in n and n.endswith('.nrrd'):
            e = int(n.split('_E0')[-1].split('.')[0])  # E0080 -> 80
            arr, _ = read_nrrd_from_zip(C.SIM_ZIP_PATH, n)
            proj[e] = arr
    return proj

def main():
    LW, LI_gt, LB, LT = load_material_projs()
    proj = load_projnoise_by_energy()
    energies = sorted(proj.keys())
    print('Found energies:', energies)

    # 1) 拟合每能量有效系数
    mu = {}
    for e in energies:
        coef = fit_mu_coeffs(LW, LI_gt, LB, LT, proj[e], sample=200000, seed=0)
        mu[e] = coef
        print(f'E={e:3d}: muW={coef[0]:.6f}, muI={coef[1]:.6f}, muB={coef[2]:.6f}, muT={coef[3]:.6f}')

    # 2) 能量对分解 + 指标
    rows = []
    for e1, e2 in C.SIM_PAIRS:
        if e1 not in proj or e2 not in proj:
            print(f'Skip pair {e1}/{e2} (missing energy)')
            continue

        muW1, muI1 = mu[e1][0], mu[e1][1]
        muW2, muI2 = mu[e2][0], mu[e2][1]
        _, LI_est = decompose_WI(proj[e1], proj[e2], muW1, muI1, muW2, muI2)

        leak = bone_leakage_metric(LI_est, LB, thresh=C.BONE_RAY_THRESH)

        rows.append({
            'pair': f'{e1}/{e2}',
            'deltaE': abs(e2 - e1),
            'iodine_RMSE_vs_GT': rmse(LI_est, LI_gt),
            'iodine_relRMSE_vs_GT': rel_rmse(LI_est, LI_gt),
            'bone_falseI_mean': leak['falseI_mean'],
            'bone_falseI_abs_mean': leak['falseI_abs_mean'],
            'bone_mask_count': leak['bone_mask_count'],
            'bone_LB_thresh': leak['lb_thresh'],
        })

    df = pd.DataFrame(rows).sort_values('deltaE')
    out_csv = C.OUT_DIR / 'sim_projection_domain_results.csv'
    save_df(df, out_csv)

    print('\\nSaved:', out_csv)
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()
