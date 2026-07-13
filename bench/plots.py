"""Benchmark plots. Consumes bench/results/{speed,accuracy}.json (produced
by bench.speed / bench.accuracy) and writes PNGs to assets/.

Figures (matplotlib, no seaborn, one chart per file):
  assets/fit_time.png   median fit seconds per (arch, dataset), grouped bars
                        per contender, log scale — one row per architecture
  assets/accuracy.png   test accuracy per (arch, dataset) per contender,
                        sklearn bar hatched + labeled "non-CNN baseline"
  assets/peak_rss.png   peak RSS per contender (import + fit), grouped bars
                        for the two representative pairs

Never invents data: exits with a message if a JSON input is missing.

Run from the repo root:  python -m bench.plots
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "bench" / "results"
ASSETS = REPO_ROOT / "assets"

# One stable color per contender across every plot (categorical slots
# validated with the dataviz six-checks script against the light surface:
# worst adjacent-pair CVD dE 13.9; the two lower-contrast hues are relieved
# by the direct value label every bar carries). torch red and tensorflow
# orange match the perceptron benchmark plots.
COLORS = {
    "ours": "#2a78d6",         # blue
    "ours_numpy": "#1baf7a",   # aqua
    "torch": "#e34948",        # red
    "tensorflow": "#d96b2f",   # orange
    "sklearn": "#eda100",      # yellow
}
LABELS = {
    "ours": "ours (C engine)",
    "ours_numpy": "ours (numpy)",
    "torch": "torch",
    "tensorflow": "tensorflow",
    "sklearn": "sklearn MLP (non-CNN)",
}
ORDER = ["ours", "ours_numpy", "torch", "tensorflow", "sklearn"]

# Opaque light surface so the PNG reads on GitHub light AND dark themes.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": AXIS,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
    })


def _load(name):
    path = RESULTS / name
    if not path.is_file():
        raise SystemExit(f"{path.relative_to(REPO_ROOT)} missing — run "
                         f"python -m bench.{name.split('.')[0]} first")
    return json.loads(path.read_text())


def _short_env(env) -> str:
    bits = [env.get("cpu", "?"), f"Python {env.get('python', '?')}"]
    if env.get("date"):
        bits.append(env["date"])
    return "  ·  ".join(bits)


def _contender_order(per_pair):
    seen = {c for pair in per_pair.values() for c in pair}
    return [c for c in ORDER if c in seen] + sorted(seen - set(ORDER))


def _grouped_bars(ax, contenders, datasets, values, log=False, fmt="{:.2f}"):
    """values[contender][dataset] -> float. One group per dataset."""
    n_series = len(contenders)
    x = np.arange(len(datasets))
    width = 0.8 / n_series
    all_h = []
    for si, c in enumerate(contenders):
        heights = [values[c].get(d, 0.0) for d in datasets]
        offset = (si - (n_series - 1) / 2) * width
        bars = ax.bar(x + offset, heights, width, label=LABELS[c],
                      color=COLORS[c], edgecolor=SURFACE, linewidth=0.8,
                      hatch="//" if c == "sklearn" else None, zorder=3)
        for rect, h in zip(bars, heights):
            if h <= 0:
                continue
            all_h.append(h)
            ax.annotate(fmt.format(h), (rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, rotation=90,
                        color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    if log:
        ax.set_yscale("log")
        # headroom for the rotated value labels + legend row above tall bars
        ax.set_ylim(min(all_h) / 3.0, max(all_h) * 14.0)
    else:
        ax.margins(y=0.18)


def _two_arch_figure(per_pair, datasets, archs, ylabel, title, env, log,
                     fmt, legend_loc, ylim=None):
    """One row of grouped bars per architecture; shared legend on top row."""
    contenders = _contender_order(per_pair)
    fig, axes = plt.subplots(len(archs), 1, figsize=(9.5, 4.1 * len(archs)),
                             dpi=150)
    for ai, (arch, ax) in enumerate(zip(archs, np.atleast_1d(axes))):
        values = {c: {d: per_pair[f"{arch}/{d}"][c] for d in datasets
                      if c in per_pair.get(f"{arch}/{d}", {})}
                  for c in contenders}
        _grouped_bars(ax, contenders, datasets, values, log=log, fmt=fmt)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if ai == 0:
            ax.set_title(f"{title} — {arch}", color=INK, fontsize=13,
                         fontweight="bold", pad=30, loc="left")
            ax.text(0, 1.04, _short_env(env), transform=ax.transAxes,
                    fontsize=8, color=INK2, va="bottom")
            ax.legend(loc=legend_loc, framealpha=0.9, facecolor=SURFACE,
                      edgecolor=GRID, fontsize=7.5, ncol=len(contenders))
        else:
            ax.set_title(arch, color=INK, fontsize=12, fontweight="bold",
                         pad=10, loc="left")
    return fig


def plot_fit_time(speed):
    datasets = speed["protocol"]["datasets"]
    archs = speed["protocol"]["architectures"]
    per_pair = {pair: {c: v["median"] for c, v in row.items()}
                for pair, row in speed["fit_s"].items()}
    r = speed["protocol"]["repeats"]
    fig = _two_arch_figure(
        per_pair, datasets, archs,
        ylabel="median fit time — s (log scale)",
        title=f"Training time — median of {r} interleaved fits",
        env=speed["env"], log=True, fmt="{:.2f}", legend_loc="upper left")
    _save(fig, "fit_time.png")


def plot_accuracy(acc):
    datasets = acc["protocol"]["datasets"]
    archs = acc["protocol"]["architectures"]
    per_pair = {pair: {c: v["test_acc"] for c, v in row.items()}
                for pair, row in acc["accuracy"].items()}
    e = acc["protocol"]["epochs"]
    n = acc["protocol"]["n_train"]
    fig = _two_arch_figure(
        per_pair, datasets, archs,
        ylabel="test accuracy",
        title=f"Test accuracy — {e} epochs on {n} samples",
        env=acc["env"], log=False, fmt="{:.3f}", legend_loc="lower right",
        ylim=(0, 1.12))
    for ax in fig.axes:
        ax.set_yticks(np.arange(0, 1.01, 0.2))
    _save(fig, "accuracy.png")


def plot_peak_rss(speed):
    """Peak RSS per contender (import + one full fit, whole-process peak),
    grouped bars for the two representative pairs. RSS is essentially flat
    across the 28x28 datasets, so lenet5/mnist and minivgg/cifar10 span the
    range (raw values for every pair stay in speed.json)."""
    pairs = ["lenet5/mnist", "minivgg/cifar10"]
    pairs = [p for p in pairs if p in speed["peak_rss_mb"]]
    contenders = _contender_order(speed["peak_rss_mb"])
    x = np.arange(len(contenders))
    width = 0.8 / len(pairs)

    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=150)
    # Bars are grouped by contender; the pair is encoded by shade (same hue).
    for pi, pair in enumerate(pairs):
        rss = speed["peak_rss_mb"][pair]
        heights = [rss.get(c, 0.0) for c in contenders]
        offset = (pi - (len(pairs) - 1) / 2) * width
        bars = ax.bar(x + offset, heights, width,
                      color=[COLORS[c] for c in contenders],
                      edgecolor=SURFACE, linewidth=0.8,
                      alpha=1.0 if pi == 0 else 0.55, zorder=3)
        for rect, h in zip(bars, heights):
            if h <= 0:
                continue
            ax.annotate(f"{h:.0f}", (rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in contenders], rotation=12,
                       ha="right", fontsize=9)
    ax.set_ylabel("peak RSS — MB")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.2)
    ax.set_title("Peak memory — import + one fit, fresh process", color=INK,
                 fontsize=13, fontweight="bold", pad=30, loc="left")
    ax.text(0, 1.05, _short_env(speed["env"]), transform=ax.transAxes,
            fontsize=8, color=INK2, va="bottom")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=INK2,
                             alpha=1.0 if pi == 0 else 0.55)
               for pi in range(len(pairs))]
    ax.legend(handles, pairs, loc="upper left", framealpha=0.9,
              facecolor=SURFACE, edgecolor=GRID, fontsize=8, ncol=len(pairs))
    _save(fig, "peak_rss.png")


def _save(fig, name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    # metadata=Software:None -> byte-stable across runs (no version/timestamp).
    fig.savefig(ASSETS / name, dpi=150, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote assets/{name}")


def main() -> int:
    _style()
    speed = _load("speed.json")
    acc = _load("accuracy.json")
    plot_fit_time(speed)
    plot_accuracy(acc)
    plot_peak_rss(speed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
