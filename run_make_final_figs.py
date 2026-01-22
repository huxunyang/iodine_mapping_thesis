import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv(Path("outputs") / "sim_domain_comparison.csv")

# ---- Fig1: relRMSE vs ΔE (two lines) ----
plt.figure()
for dom in ["projection", "image"]:
    d = df[df["domain"] == dom].sort_values("deltaE")
    plt.plot(d["deltaE"], d["iodine_relRMSE_vs_GT"], marker="o", label=dom)

plt.xlabel("ΔE")
plt.ylabel("Iodine relRMSE vs GT")
plt.title("Simulation: relRMSE vs ΔE (Projection vs Image domain)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(Path("outputs") / "final_relRMSE_vs_deltaE_domains.png", dpi=200)
plt.close()

# ---- Fig2: relRMSE vs cond(A) (two scatters) ----
plt.figure()
for dom in ["projection", "image"]:
    d = df[df["domain"] == dom]
    plt.scatter(d["cond_A"], d["iodine_relRMSE_vs_GT"], label=dom)

    for _, r in d.iterrows():
        plt.annotate(r["pair"], (r["cond_A"], r["iodine_relRMSE_vs_GT"]))

plt.xlabel("cond(A)")
plt.ylabel("Iodine relRMSE vs GT")
plt.title("Simulation: relRMSE vs cond(A) (Projection vs Image domain)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(Path("outputs") / "final_relRMSE_vs_condA_domains.png", dpi=200)
plt.close()

print("Saved:")
print(" - outputs/final_relRMSE_vs_deltaE_domains.png")
print(" - outputs/final_relRMSE_vs_condA_domains.png")
