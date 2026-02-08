import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv(Path("outputs") / "sim_domain_comparison.csv")

styles = {
    "projection": dict(marker="o", linestyle="-"),
    "image": dict(marker="s", linestyle="--"),
}

# ---- Fig1: relRMSE vs ΔE ----
plt.figure()
for dom in ["projection", "image"]:
    d = df[df["domain"] == dom].sort_values("deltaE")
    plt.plot(
        d["deltaE"],
        d["iodine_relRMSE_vs_GT"],
        label=dom,
        **styles[dom]
    )

plt.xlabel("ΔE")
plt.ylabel("Iodine relRMSE vs GT")
plt.title("Simulation study: relRMSE vs ΔE\nProjection-domain vs Image-domain")
plt.xticks(sorted(df["deltaE"].unique()))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(Path("outputs") / "final_relRMSE_vs_deltaE_domains.png", dpi=300)
plt.close()

# ---- Fig2: relRMSE vs cond(A) ----
plt.figure()
for dom in ["projection", "image"]:
    d = df[df["domain"] == dom]
    plt.scatter(
        d["cond_A"],
        d["iodine_relRMSE_vs_GT"],
        label=dom,
        marker=styles[dom]["marker"]
    )

    if dom == "projection":
        for _, r in d.iterrows():
            plt.annotate(
                r["pair"],
                (r["cond_A"], r["iodine_relRMSE_vs_GT"]),
                fontsize=9
            )

plt.xlabel("cond(A)")
plt.ylabel("Iodine relRMSE vs GT")
plt.title("Simulation study: relRMSE vs cond(A)\nProjection-domain vs Image-domain")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(Path("outputs") / "final_relRMSE_vs_condA_domains.png", dpi=300)
plt.close()

print("Saved final comparison figures.")
