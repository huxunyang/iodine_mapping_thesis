'''
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("outputs")
OUT.mkdir(exist_ok=True, parents=True)

BASE_CSV = OUT / "real_results_pairwise.csv"
WI_CSV   = OUT / "real_wi_results_pairwise.csv"
WI_COEF  = OUT / "real_wi_coeffs.csv"

# ---- options ----
EXCLUDE_ZERO_FOR_AGG = True     # 聚合RMSE/|bias|时是否排除0 mg/mL
WEIGHT_BY_NREP = True          # 聚合时是否按n_rep加权
SAVE_TABLE = True


def parse_pair(pair_str: str):
    a, b = pair_str.split("/")
    e1, e2 = int(a), int(b)
    return e1, e2, abs(e2 - e1)


def wmean(x, w=None):
    x = np.asarray(x, dtype=float)
    if w is None:
        return float(np.mean(x))
    w = np.asarray(w, dtype=float)
    s = float(np.sum(w))
    if s <= 0:
        return float(np.mean(x))
    return float(np.sum(w * x) / s)


def load_method(csv_path: Path, method_name: str):
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    # normalize column names
    required = {"pair","conc_true_mgml","n_rep","pred_mean","bias","rmse"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"{csv_path.name} missing columns: {miss}")

    e = df["pair"].apply(parse_pair)
    df["E1"] = [t[0] for t in e]
    df["E2"] = [t[1] for t in e]
    df["deltaE"] = [t[2] for t in e]
    df["method"] = method_name
    return df


def aggregate_metrics(df: pd.DataFrame):
    d = df.copy()
    if EXCLUDE_ZERO_FOR_AGG:
        d = d[d["conc_true_mgml"] > 0].copy()

    rows = []
    for (method, pair), g in d.groupby(["method","pair"]):
        w = g["n_rep"] if WEIGHT_BY_NREP else None
        rmse_agg = wmean(g["rmse"], w)
        absbias_agg = wmean(np.abs(g["bias"]), w)
        deltaE = int(g["deltaE"].iloc[0])
        rows.append({
            "method": method,
            "pair": pair,
            "deltaE": deltaE,
            "RMSE_agg": rmse_agg,
            "AbsBias_agg": absbias_agg,
        })

    agg = pd.DataFrame(rows).sort_values(["deltaE","pair","method"]).reset_index(drop=True)
    return agg


def attach_wi_condA(agg: pd.DataFrame):
    if not WI_COEF.exists():
        return agg
    c = pd.read_csv(WI_COEF)
    if "pair" not in c.columns or "cond_A" not in c.columns:
        return agg
    c = c[["pair","cond_A","det_A"]].copy()
    # merge only for WI rows
    out = agg.copy()
    out = out.merge(c, on="pair", how="left")
    return out


def fig_rmse_vs_deltaE(agg: pd.DataFrame):
    plt.figure()
    for method, g in agg.groupby("method"):
        # jitter x a tiny bit so two methods don't overlap perfectly
        x = g["deltaE"].to_numpy(dtype=float)
        if method.lower().startswith("wi"):
            x = x + 0.3
        else:
            x = x - 0.3
        plt.scatter(x, g["RMSE_agg"], label=method)
        for xi, yi, pair in zip(x, g["RMSE_agg"], g["pair"]):
            plt.annotate(pair, (xi, yi))

    zflag = "exclude 0" if EXCLUDE_ZERO_FOR_AGG else "include 0"
    wflag = "weighted" if WEIGHT_BY_NREP else "unweighted"
    plt.xlabel("ΔE (kVp difference)")
    plt.ylabel("Aggregated RMSE (mg/mL)")
    plt.title(f"Real: RMSE vs ΔE (Baseline vs WI, {zflag}, {wflag})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = OUT / "real_cmp_F1_rmse_vs_deltaE.png"
    plt.savefig(out, dpi=250)
    plt.close()
    return out


def fig_absbias_vs_deltaE(agg: pd.DataFrame):
    plt.figure()
    for method, g in agg.groupby("method"):
        x = g["deltaE"].to_numpy(dtype=float)
        if method.lower().startswith("wi"):
            x = x + 0.3
        else:
            x = x - 0.3
        plt.scatter(x, g["AbsBias_agg"], label=method)
        for xi, yi, pair in zip(x, g["AbsBias_agg"], g["pair"]):
            plt.annotate(pair, (xi, yi))

    zflag = "exclude 0" if EXCLUDE_ZERO_FOR_AGG else "include 0"
    wflag = "weighted" if WEIGHT_BY_NREP else "unweighted"
    plt.xlabel("ΔE (kVp difference)")
    plt.ylabel("Aggregated |Bias| (mg/mL)")
    plt.title(f"Real: |Bias| vs ΔE (Baseline vs WI, {zflag}, {wflag})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = OUT / "real_cmp_F2_absbias_vs_deltaE.png"
    plt.savefig(out, dpi=250)
    plt.close()
    return out


def fig_bias_vs_conc_allpairs(df_base: pd.DataFrame, df_wi: pd.DataFrame):
    """
    One figure: for each pair, plot bias vs concentration with two lines:
      - baseline (marker o)
      - WI (marker x)
    """
    plt.figure()

    pairs = sorted(set(df_base["pair"]) | set(df_wi["pair"]))
    for pair in pairs:
        b = df_base[df_base["pair"] == pair].sort_values("conc_true_mgml")
        w = df_wi[df_wi["pair"] == pair].sort_values("conc_true_mgml")

        if not b.empty:
            plt.plot(b["conc_true_mgml"], b["bias"], marker="o", label=f"{pair} baseline")
        if not w.empty:
            plt.plot(w["conc_true_mgml"], w["bias"], marker="x", linestyle="--", label=f"{pair} WI")

    plt.axhline(0, linewidth=1)
    plt.xlabel("True iodine concentration (mg/mL)")
    plt.ylabel("Bias = Pred_mean - True (mg/mL)")
    plt.title("Real: Bias vs concentration (Baseline vs WI, per energy pair)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=1)
    plt.tight_layout()
    out = OUT / "real_cmp_F3_bias_vs_conc_pairs.png"
    plt.savefig(out, dpi=250)
    plt.close()
    return out


def main():
    df_base = load_method(BASE_CSV, "baseline")
    df_wi   = load_method(WI_CSV, "WI")

    agg = aggregate_metrics(pd.concat([df_base, df_wi], ignore_index=True))
    agg2 = attach_wi_condA(agg)

    if SAVE_TABLE:
        agg2.to_csv(OUT / "real_cmp_summary.csv", index=False)

    o1 = fig_rmse_vs_deltaE(agg)
    o2 = fig_absbias_vs_deltaE(agg)
    o3 = fig_bias_vs_conc_allpairs(df_base, df_wi)

    # console summary (useful for thesis text)
    print("=== Aggregated metrics (per pair) ===")
    print(agg2.to_string(index=False))

    print("\nSaved figures:")
    print(" -", o1)
    print(" -", o2)
    print(" -", o3)
    if SAVE_TABLE:
        print("Saved table:")
        print(" -", OUT / "real_cmp_summary.csv")


if __name__ == "__main__":
    main()
'''
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("outputs")
OUT.mkdir(exist_ok=True, parents=True)

BASE_CSV = OUT / "real_results_pairwise.csv"
WI_CSV   = OUT / "real_wi_results_pairwise.csv"
WI_COEF  = OUT / "real_wi_coeffs.csv"

# ---- options ----
EXCLUDE_ZERO_FOR_AGG = True
WEIGHT_BY_NREP = True
SAVE_TABLE = True


def parse_pair(pair_str: str):
    a, b = pair_str.split("/")
    e1, e2 = int(a), int(b)
    return e1, e2, abs(e2 - e1)


def wmean(x, w=None):
    x = np.asarray(x, dtype=float)
    if w is None:
        return float(np.mean(x))
    w = np.asarray(w, dtype=float)
    s = float(np.sum(w))
    if s <= 0:
        return float(np.mean(x))
    return float(np.sum(w * x) / s)


def load_method(csv_path: Path, method_name: str):
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    required = {"pair","conc_true_mgml","n_rep","pred_mean","bias","rmse"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"{csv_path.name} missing columns: {miss}")

    e = df["pair"].apply(parse_pair)
    df["E1"] = [t[0] for t in e]
    df["E2"] = [t[1] for t in e]
    df["deltaE"] = [t[2] for t in e]
    df["method"] = method_name
    return df


def aggregate_metrics(df: pd.DataFrame):
    d = df.copy()
    if EXCLUDE_ZERO_FOR_AGG:
        d = d[d["conc_true_mgml"] > 0].copy()

    rows = []
    for (method, pair), g in d.groupby(["method","pair"]):
        w = g["n_rep"] if WEIGHT_BY_NREP else None
        rmse_agg = wmean(g["rmse"], w)
        absbias_agg = wmean(np.abs(g["bias"]), w)
        deltaE = int(g["deltaE"].iloc[0])
        rows.append({
            "method": method,
            "pair": pair,
            "deltaE": deltaE,
            "RMSE_agg": rmse_agg,
            "AbsBias_agg": absbias_agg,
        })

    agg = pd.DataFrame(rows).sort_values(["deltaE","pair","method"]).reset_index(drop=True)
    return agg


def attach_wi_condA(agg: pd.DataFrame):
    if not WI_COEF.exists():
        return agg
    c = pd.read_csv(WI_COEF)
    if "pair" not in c.columns or "cond_A" not in c.columns:
        return agg
    keep_cols = ["pair", "cond_A"]
    if "det_A" in c.columns:
        keep_cols.append("det_A")
    c = c[keep_cols].copy()
    out = agg.copy()
    out = out.merge(c, on="pair", how="left")
    return out


def annotate_pair_points(ax, xvals, yvals, pairs, method_name):
    """
    给 F1/F2 的点做防重叠标注。
    通过 pair + method 指定不同偏移。
    """
    # 你可以按实际效果再微调这些 offset
    offset_map = {
        ("baseline", "80/100"): (-18,  10),
        ("baseline", "100/120"): (  8, -14),
        ("baseline", "80/120"): (-10,  10),
        ("wi",       "80/100"): (-18, -14),
        ("wi",       "100/120"): (  8,  10),
        ("wi",       "80/120"): (  8, -14),
    }

    m = method_name.strip().lower()
    for x, y, pair in zip(xvals, yvals, pairs):
        dx, dy = offset_map.get((m, pair), (6, 6))
        ax.annotate(
            pair,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75, ec="none")
        )


def fig_rmse_vs_deltaE(agg: pd.DataFrame):
    fig, ax = plt.subplots()

    for method, g in agg.groupby("method"):
        x = g["deltaE"].to_numpy(dtype=float)

        # baseline / WI 左右错开一点
        if method.lower().startswith("wi"):
            x = x + 0.35
        else:
            x = x - 0.35

        y = g["RMSE_agg"].to_numpy(dtype=float)
        pairs = g["pair"].tolist()

        ax.scatter(x, y, label=method, s=45)
        annotate_pair_points(ax, x, y, pairs, method)

    zflag = "exclude 0" if EXCLUDE_ZERO_FOR_AGG else "include 0"
    wflag = "weighted" if WEIGHT_BY_NREP else "unweighted"
    ax.set_xlabel("ΔE (kVp difference)")
    ax.set_ylabel("Aggregated RMSE (mg/mL)")
    ax.set_title(f"Real: RMSE vs ΔE (Baseline vs WI, {zflag}, {wflag})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = OUT / "real_cmp_F1_rmse_vs_deltaE.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out


def fig_absbias_vs_deltaE(agg: pd.DataFrame):
    fig, ax = plt.subplots()

    for method, g in agg.groupby("method"):
        x = g["deltaE"].to_numpy(dtype=float)

        if method.lower().startswith("wi"):
            x = x + 0.35
        else:
            x = x - 0.35

        y = g["AbsBias_agg"].to_numpy(dtype=float)
        pairs = g["pair"].tolist()

        ax.scatter(x, y, label=method, s=45)
        annotate_pair_points(ax, x, y, pairs, method)

    zflag = "exclude 0" if EXCLUDE_ZERO_FOR_AGG else "include 0"
    wflag = "weighted" if WEIGHT_BY_NREP else "unweighted"
    ax.set_xlabel("ΔE (kVp difference)")
    ax.set_ylabel("Aggregated |Bias| (mg/mL)")
    ax.set_title(f"Real: |Bias| vs ΔE (Baseline vs WI, {zflag}, {wflag})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = OUT / "real_cmp_F2_absbias_vs_deltaE.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out


def fig_bias_vs_conc_allpairs(df_base: pd.DataFrame, df_wi: pd.DataFrame):
    fig, ax = plt.subplots()

    pairs = sorted(set(df_base["pair"]) | set(df_wi["pair"]))
    for pair in pairs:
        b = df_base[df_base["pair"] == pair].sort_values("conc_true_mgml")
        w = df_wi[df_wi["pair"] == pair].sort_values("conc_true_mgml")

        if not b.empty:
            ax.plot(b["conc_true_mgml"], b["bias"], marker="o", label=f"{pair} baseline")
        if not w.empty:
            ax.plot(w["conc_true_mgml"], w["bias"], marker="x", linestyle="--", label=f"{pair} WI")

    ax.axhline(0, linewidth=1)
    ax.set_xlabel("True iodine concentration (mg/mL)")
    ax.set_ylabel("Bias = Pred_mean - True (mg/mL)")
    ax.set_title("Real: Bias vs concentration (Baseline vs WI, per energy pair)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=1)
    fig.tight_layout()

    out = OUT / "real_cmp_F3_bias_vs_conc_pairs.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out


def main():
    df_base = load_method(BASE_CSV, "baseline")
    df_wi   = load_method(WI_CSV, "WI")

    agg = aggregate_metrics(pd.concat([df_base, df_wi], ignore_index=True))
    agg2 = attach_wi_condA(agg)

    if SAVE_TABLE:
        agg2.to_csv(OUT / "real_cmp_summary.csv", index=False)

    o1 = fig_rmse_vs_deltaE(agg)
    o2 = fig_absbias_vs_deltaE(agg)
    o3 = fig_bias_vs_conc_allpairs(df_base, df_wi)

    print("=== Aggregated metrics (per pair) ===")
    print(agg2.to_string(index=False))

    print("\nSaved figures:")
    print(" -", o1)
    print(" -", o2)
    print(" -", o3)
    if SAVE_TABLE:
        print("Saved table:")
        print(" -", OUT / "real_cmp_summary.csv")


if __name__ == "__main__":
    main()