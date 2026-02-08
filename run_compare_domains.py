import pandas as pd
from pathlib import Path

out = Path("outputs")

proj = pd.read_csv(out / "sim_projection_domain_results.csv")
img  = pd.read_csv(out / "sim_image_domain_results.csv")

proj["domain"] = "projection"
img["domain"]  = "image"

keep = ["pair","deltaE","cond_A","iodine_relRMSE_vs_GT","bone_falseI_abs_mean","domain"]

# 列检查
for name, df_ in [("projection", proj), ("image", img)]:
    missing = set(keep) - set(df_.columns)
    if missing:
        raise KeyError(f"{name} CSV missing columns: {missing}")

proj = proj[keep]
img  = img[keep]

df = pd.concat([proj, img], ignore_index=True)
df["domain"] = pd.Categorical(df["domain"], categories=["projection", "image"])

df.to_csv(out / "sim_domain_comparison.csv", index=False)

print(df.sort_values(["pair","domain"]).to_string(index=False))
