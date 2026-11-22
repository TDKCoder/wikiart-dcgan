"""Report figures from runs/<name>/losses.csv and metrics.jsonl.

python plots.py dcgan64 [other_run ...] --out report/figs [--fmt png] [--labels name1 name2 ...]
Writes, in --fmt:
  losses_<run>, dlogits_<run>  per-run G/D loss and mean D output curves
  fid                          FID vs epoch for every run that has metrics
  training                     two panels for the report: (a) losses of the first run, (b) FID vs epoch
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical slots from the dataviz reference palette, assigned in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
SKIP = 300  # first iterations have huge transients that would set the y-range

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "lines.linewidth": 1.2, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def plot_pair(ax, df, cols, labels, ylabel, smooth=300):
    df = df[df["step"] > SKIP]
    x = df["step"] / 1000
    ends = []
    for i, (c, lab) in enumerate(zip(cols, labels)):
        ax.plot(x, df[c], color=SERIES[i], alpha=0.15, linewidth=0.5)
        s = df[c].ewm(span=smooth).mean()
        ax.plot(x, s, color=SERIES[i], label=lab)
        ends.append(s.iloc[-1])
    lo, hi = np.percentile(df[cols].values, [0.5, 99.5])
    pad = 0.05 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    # Direct end labels only when they cannot collide; the legend always carries identity.
    if abs(ends[0] - ends[1]) > 0.08 * (hi - lo):
        for lab, e in zip(labels, ends):
            ax.annotate(lab, (x.iloc[-1], e), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=7, color=INK, annotation_clip=False)
    ax.set_xlabel("iteration (k)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.2)


def plot_fid(ax, fid_runs, labels):
    for i, (run, m) in enumerate(fid_runs):
        ax.plot(m["epoch"], m["fid"], color=SERIES[i], marker="o", markersize=3, label=labels.get(run, run))
    ax.set_xlabel("epoch")
    ax.set_ylabel("FID (lower is better)")
    if len(fid_runs) > 1:
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(fid_runs), borderaxespad=0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="report/figs")
    ap.add_argument("--which", default="ema")
    ap.add_argument("--fmt", default="pdf", help="pdf for LaTeX, png for quick viewing")
    ap.add_argument("--labels", nargs="*", default=[], help="display names, one per run")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = dict(zip(args.runs, args.labels))

    fid_runs, losses = [], {}
    for run in args.runs:
        d = Path("runs") / run
        df = pd.read_csv(d / "losses.csv")
        losses[run] = df
        for cols, labs, ylabel, name in [(["loss_d", "loss_g"], ["D loss", "G loss"], "loss", "losses"),
                                         (["d_real", "d_fake"], ["D(real)", "D(fake)"], "mean D logit", "dlogits")]:
            fig, ax = plt.subplots(figsize=(3.3, 2.0))
            plot_pair(ax, df, cols, labs, ylabel)
            fig.savefig(out / f"{name}_{run}.{args.fmt}")
            plt.close(fig)
        if (d / "metrics.jsonl").exists():
            m = pd.DataFrame([json.loads(line) for line in open(d / "metrics.jsonl")])
            m = m[m["which"] == args.which].drop_duplicates("epoch", keep="last").sort_values("epoch")
            if len(m):
                fid_runs.append((run, m))

    if fid_runs:
        fig, ax = plt.subplots(figsize=(3.3, 2.0))
        plot_fid(ax, fid_runs, labels)
        fig.savefig(out / f"fid.{args.fmt}")
        plt.close(fig)

        fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 1.75), gridspec_kw={"wspace": 0.45})
        plot_pair(a, losses[args.runs[0]], ["loss_d", "loss_g"], ["D loss", "G loss"], "hinge loss")
        a.set_title("(a)", loc="left")
        plot_fid(b, fid_runs, labels)
        b.set_title("(b)", loc="left")
        fig.savefig(out / f"training.{args.fmt}")
        plt.close(fig)
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()
