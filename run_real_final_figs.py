from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- paths ----
CSV = Path("outputs") / "real_results_pairwise.csv"
OUT = Path("outputs")
OUT.mkdir(exist_ok=True, parents=True)

# ---- settings ----
EXCLUDE_ZERO_CONC_FOR_AGG = True   # Fig R1 聚合RMSE是否排除0 mg/mL
USE_WEIGHT_BY_NREP = True         # Fig R1 聚合时是否按 n_rep 加权
TITLE_PREFIX = "Real (baseline regression)"


def parse_pair(pair_str: str):
    a, b = pair_str.split("/")
    e1, e2 = int(a), int(b)
    return e1, e2, abs(e2 - e1)


def weighted_mean(x, w=None):
    x = np.asarray(x, dtype=float)
    if w is None:
        return float(np.mean(x))
    w = np.asarray(w, dtype=float)
    if np.sum(w) <= 0:
        return float(np.mean(x))
    return float(np.sum(w * x) / np.sum(w))


def load():
    if not CSV.exists():
        raise FileNotFoundError(f"Missing: {CSV.resolve()}")
    df = pd.read_csv(CSV)
    # add energies and deltaE
    e = df["pair"].apply(parse_pair)
    df["E1"] = [t[0] for t in e]
    df["E2"] = [t[1] for t in e]
    df["deltaE"] = [t[2] for t in e]
    return df


def fig_rmse_vs_deltaE(df: pd.DataFrame):
    d = df.copy()
    if EXCLUDE_ZERO_CONC_FOR_AGG:
        d = d[d["conc_true_mgml"] > 0].copy()

    rows = []
    for pair, g in d.groupby("pair"):
        w = g["n_rep"] if USE_WEIGHT_BY_NREP else None
        rmse = weighted_mean(g["rmse"], w)
        bias_abs = weighted_mean(np.abs(g["bias"]), w)
        rows.append({
            "pair": pair,
            "deltaE": int(g["deltaE"].iloc[0]),
            "RMSE_agg": rmse,
            "AbsBias_agg": bias_abs
        })
    s = pd.DataFrame(rows).sort_values(["deltaE", "pair"]).reset_index(drop=True)

    # Plot RMSE vs deltaE
    plt.figure()
    plt.scatter(s["deltaE"], s["RMSE_agg"])
    for _, r in s.iterrows():
        plt.annotate(r["pair"], (r["deltaE"], r["RMSE_agg"]))
    plt.xlabel("ΔE (kVp difference)")
    plt.ylabel("Aggregated RMSE (mg/mL)")
    zflag = "exclude 0" if EXCLUDE_ZERO_CONC_FOR_AGG else "include 0"
    wflag = "weighted by n_rep" if USE_WEIGHT_BY_NREP else "unweighted"
    plt.title(f"{TITLE_PREFIX}: RMSE vs ΔE ({zflag}, {wflag})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUT / "real_fig_R1_rmse_vs_deltaE.png"
    plt.savefig(out, dpi=250)
    plt.close()

    # Optional: also save the aggregated table
    s.to_csv(OUT / "real_fig_R1_rmse_vs_deltaE_table.csv", index=False)

    return out


def fig_bias_vs_conc(df: pd.DataFrame):
    # Bias vs concentration curves per pair
    plt.figure()
    for pair, g in df.groupby("pair"):
        g = g.sort_values("conc_true_mgml")
        plt.plot(g["conc_true_mgml"], g["bias"], marker="o", label=pair)

    plt.axhline(0, linewidth=1)
    plt.xlabel("True iodine concentration (mg/mL)")
    plt.ylabel("Bias = Pred_mean - True (mg/mL)")
    plt.title(f"{TITLE_PREFIX}: Bias vs concentration (per energy pair)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = OUT / "real_fig_R2_bias_vs_conc.png"
    plt.savefig(out, dpi=250)
    plt.close()
    return out


def fig_pred_vs_true(df: pd.DataFrame):
    # Scatter predicted mean vs true for all pairs
    plt.figure()
    for pair, g in df.groupby("pair"):
        plt.scatter(g["conc_true_mgml"], g["pred_mean"], label=pair)

    # y=x reference line
    xmin = float(df["conc_true_mgml"].min())
    xmax = float(df["conc_true_mgml"].max())
    xs = np.linspace(xmin, xmax, 200)
    plt.plot(xs, xs, linewidth=1)

    plt.xlabel("True iodine concentration (mg/mL)")
    plt.ylabel("Predicted mean (mg/mL)")
    plt.title(f"{TITLE_PREFIX}: Predicted vs true (mean over reps)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = OUT / "real_fig_R3_pred_vs_true.png"
    plt.savefig(out, dpi=250)
    plt.close()
    return out


def main():
    df = load()

    # Print a compact summary to console (useful for writing Results)
    # Aggregated RMSE per pair (exclude 0 by default)
    d = df[df["conc_true_mgml"] > 0].copy() if EXCLUDE_ZERO_CONC_FOR_AGG else df.copy()
    for pair, g in d.groupby("pair"):
        rmse = weighted_mean(g["rmse"], g["n_rep"] if USE_WEIGHT_BY_NREP else None)
        ab = weighted_mean(np.abs(g["bias"]), g["n_rep"] if USE_WEIGHT_BY_NREP else None)
        print(f"{pair}: deltaE={int(g['deltaE'].iloc[0])}, RMSE_agg={rmse:.3f}, absBias_agg={ab:.3f}")

    o1 = fig_rmse_vs_deltaE(df)
    o2 = fig_bias_vs_conc(df)
    o3 = fig_pred_vs_true(df)

    print("Saved figures:")
    print(" -", o1)
    print(" -", o2)
    print(" -", o3)
    print("Saved table:")
    print(" -", OUT / "real_fig_R1_rmse_vs_deltaE_table.csv")


if __name__ == "__main__":
    main()
