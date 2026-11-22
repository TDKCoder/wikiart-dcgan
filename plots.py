"""Report figures from runs/<name>/losses.csv and metrics.jsonl.

python plots.py dcgan64 [other_run ...] --out report/figs
Writes losses_<run>, dlogits_<run> and, when metrics exist, fid (all runs) in --fmt (pdf or png).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical slots from the dataviz reference palette, assigned in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "lines.linewidth": 1.2, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def two_series(df, cols, labels, ylabel, path, smooth=200):
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    x = df["step"] / 1000
    for i, (c, lab) in enumerate(zip(cols, labels)):
        ax.plot(x, df[c], color=SERIES[i], alpha=0.15, linewidth=0.5)
        s = df[c].ewm(span=smooth).mean()
        ax.plot(x, s, color=SERIES[i], label=lab)
        ax.annotate(lab, (x.iloc[-1], s.iloc[-1]), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK)
    ax.set_xlabel("iteration (k)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="report/figs")
    ap.add_argument("--which", default="ema")
    ap.add_argument("--fmt", default="pdf", help="pdf for LaTeX, png for quick viewing")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fid_runs = []
    for run in args.runs:
        d = Path("runs") / run
        df = pd.read_csv(d / "losses.csv")
        two_series(df, ["loss_d", "loss_g"], ["D loss", "G loss"], "loss", out / f"losses_{run}.{args.fmt}")
        two_series(df, ["d_real", "d_fake"], ["D(real)", "D(fake)"], "mean D logit", out / f"dlogits_{run}.{args.fmt}")
        if (d / "metrics.jsonl").exists():
            m = pd.DataFrame([json.loads(line) for line in open(d / "metrics.jsonl")])
            m = m[m["which"] == args.which].drop_duplicates("epoch", keep="last").sort_values("epoch")
            if len(m):
                fid_runs.append((run, m))

    if fid_runs:
        fig, ax = plt.subplots(figsize=(3.3, 2.0))
        for i, (run, m) in enumerate(fid_runs):
            ax.plot(m["epoch"], m["fid"], color=SERIES[i], marker="o", markersize=3, label=run)
        ax.set_xlabel("epoch")
        ax.set_ylabel("FID (lower is better)")
        if len(fid_runs) > 1:
            ax.legend()
        fig.savefig(out / f"fid.{args.fmt}")
        plt.close(fig)
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()
