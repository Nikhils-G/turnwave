"""Plot training curves from a run's log.csv.

    python scripts/plot_training.py checkpoints/text_eot/log.csv --out docs/training_curves.png

Two panels rather than one with twin y-axes: loss and average precision live on
different scales, and a dual-axis chart lets the reader infer a relationship from
whatever the axis scaling happens to imply. Kept side by side, the divergence
between them is the actual finding — validation loss climbs while AP keeps
improving for another ~1500 steps.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange


def read_log(path: Path) -> dict[str, list[float]]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {key: [float(row[key]) for row in rows] for key in rows[0]}


def style_axis(ax, title: str, ylabel: str):
    ax.set_title(title, color=INK, fontsize=11, pad=10, loc="left")
    ax.set_xlabel("training step", color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path, nargs="?", default=Path("checkpoints/text_eot/log.csv"))
    ap.add_argument("--out", type=Path, default=Path("docs/training_curves.png"))
    args = ap.parse_args(argv)

    log = read_log(args.log)
    steps = log["step"]
    best_i = max(range(len(steps)), key=lambda i: log["val_ap"][i])
    best_step, best_ap = steps[best_i], log["val_ap"][best_i]

    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)

    ax_loss.plot(steps, log["train_loss"], color=SERIES_1, linewidth=2, label="train", zorder=3)
    ax_loss.plot(steps, log["val_loss"], color=SERIES_2, linewidth=2, label="validation", zorder=3)
    style_axis(ax_loss, "Loss — the model starts memorizing", "binary cross-entropy")

    ax_metric.plot(steps, log["val_ap"], color=SERIES_1, linewidth=2,
                   label="average precision", zorder=3)
    ax_metric.plot(steps, log["val_f1"], color=SERIES_2, linewidth=2, label="F1", zorder=3)
    ax_metric.plot([best_step], [best_ap], marker="o", markersize=8, color=SERIES_1,
                   markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    ax_metric.annotate(
        f"checkpoint kept\nstep {best_step:,.0f} · AP {best_ap:.3f}",
        xy=(best_step, best_ap), xytext=(-10, -46), textcoords="offset points",
        color=INK_MUTED, fontsize=8.5, ha="center",
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.8),
    )
    style_axis(ax_metric, "Validation quality — ranking holds up longer", "score")

    for ax in (ax_loss, ax_metric):
        legend = ax.legend(frameon=False, fontsize=9, loc="best")
        for text in legend.get_texts():
            text.set_color(INK_MUTED)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out}  (best AP {best_ap:.4f} at step {best_step:,.0f})")


if __name__ == "__main__":
    main()
