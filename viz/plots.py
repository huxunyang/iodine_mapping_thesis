import matplotlib.pyplot as plt

def plot_metric_vs_deltaE(df, y, out_path, title=None):
    df = df.sort_values("deltaE")
    plt.figure()
    plt.plot(df["deltaE"], df[y], marker="o")
    plt.xlabel("ΔE")
    plt.ylabel(y)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def scatter_x_y(df, x, y, out_path, title=None, annotate_col=None):
    plt.figure()
    plt.scatter(df[x], df[y])
    if annotate_col:
        for _, r in df.iterrows():
            plt.annotate(str(r[annotate_col]), (r[x], r[y]))
    plt.xlabel(x)
    plt.ylabel(y)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
