import pandas as pd
from pathlib import Path

proj = pd.read_csv(Path("outputs") / "sim_projection_domain_results.csv")
img  = pd.read_csv(Path("outputs") / "sim_image_domain_results.csv")

proj["domain"] = "projection"
img["domain"]  = "image"

# 统一列名
keep = ["pair","deltaE","cond_A","iodine_relRMSE_vs_GT","bone_falseI_abs_mean","domain"]
proj = proj[keep]
img  = img[keep]

df = pd.concat([proj, img], ignore_index=True)
df.to_csv(Path("outputs") / "sim_domain_comparison.csv", index=False)

print(df.sort_values(["pair","domain"]).to_string(index=False))
