"""Per-NNP training-loss curves for an ArcaNN training iteration.

Copied unconditionally into each NNN-training/ folder by
training/prepare.py. Run it there once the training jobs finish:

    python plot_loss.py

Handles both engines per NNP:
  * DeePMD  -> <nnp>/lcurve.out            (whitespace table, `dp train`)
  * MACE    -> <nnp>/results/*.txt         (JSON lines, mace_run_train)

For MACE the panel shows the per-epoch validation E and F RMSE (eval rows
only, converted to meV) on twin axes, plus a dashed line for the deployed
SWA stage-two model's valid RMSE parsed from <nnp>/training.log -- the JSON
eval rows only ever cover stage one, so the shipped model is 10-30x better
on energy than the last plotted point.

If the centroid (electron-position) model has run this iteration, its
per-epoch history (centroid/centroid_<iter>_history.json, written by
train_centroid.py) gets an extra panel on the right.

Writes loss.png next to this script.
"""

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["r", "b", "k", "g", "m", "c"]


def nnp_count():
    cfg = Path("../control/config.json")
    if cfg.is_file():
        return int(json.loads(cfg.read_text())["nnp_count"])
    # fall back to counting numeric sub-folders
    return sum(1 for p in Path(".").iterdir() if p.is_dir() and p.name.isdigit())


def plot_deepmd(ax, lcurve):
    data = np.genfromtxt(lcurve, names=True)
    for j, name in enumerate(data.dtype.names[1:-1]):  # skip step and lr
        ax.plot(data["step"], data[name], label=name, color=COLORS[j % len(COLORS)])
    ax.set_xlabel("step")


def mace_stage_two_valid(nnp_dir):
    """Deployed-model valid (E meV/atom, F meV/A) from the last error table
    in training.log, or None. That table is the SWA stage-two model; the
    JSON eval rows only ever cover stage one."""
    log = Path(nnp_dir) / "training.log"
    if not log.is_file():
        return None
    valid_rows = [ln for ln in log.read_text().splitlines() if "| valid_Default |" in ln]
    if not valid_rows:
        return None
    cells = [c.strip() for c in valid_rows[-1].strip().strip("|").split("|")]
    return float(cells[1]), float(cells[2])


def plot_mace(ax, results_txt, nnp_dir):
    rows = [json.loads(line) for line in Path(results_txt).read_text().splitlines() if line.strip()]
    evals = [r for r in rows if r.get("mode") == "eval" and r.get("epoch") is not None]
    if not evals:
        return
    epochs = [r["epoch"] for r in evals]
    line_e, = ax.plot(epochs, [r["rmse_e_per_atom"] * 1000 for r in evals],
                      color=COLORS[0], label="E RMSE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("E RMSE (meV/atom)", color=COLORS[0])
    ax.set_yscale("log")
    ax.tick_params(axis="y", labelcolor=COLORS[0])

    ax_f = ax.twinx()
    line_f, = ax_f.plot(epochs, [r["rmse_f"] * 1000 for r in evals],
                        color=COLORS[1], label="F RMSE")
    ax_f.set_ylabel(r"F RMSE (meV/$\AA$)", color=COLORS[1])
    ax_f.set_yscale("log")
    ax_f.tick_params(axis="y", labelcolor=COLORS[1])

    handles = [line_e, line_f]
    final = mace_stage_two_valid(nnp_dir)
    if final:
        handles.append(ax.axhline(final[0], color=COLORS[0], linestyle="--", linewidth=0.8,
                                  label=f"stage-two valid E {final[0]:.1f}"))
        handles.append(ax_f.axhline(final[1], color=COLORS[1], linestyle="--", linewidth=0.8,
                                    label=f"stage-two valid F {final[1]:.1f}"))
    ax.legend(handles=handles, loc="upper right", fontsize=7)


def plot_centroid(ax, history_json):
    history = json.loads(Path(history_json).read_text())
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    for j, key in enumerate(("train_rmse", "valid_rmse")):
        ax.plot(epochs, [h[key] for h in history], label=key, color=COLORS[j % len(COLORS)])
    ax_lr = ax.twinx()
    ax_lr.plot(epochs, [h["lr"] for h in history], color=COLORS[3], linestyle=":", label="lr")
    ax_lr.set_yscale("log")
    ax_lr.set_ylabel("learning rate")
    ax.set_xlabel("epoch")
    handles = ax.get_lines() + ax_lr.get_lines()
    ax.legend(handles, [h.get_label() for h in handles], loc="upper right")


def main():
    n = nnp_count()
    centroid_hist = sorted(glob.glob("centroid/centroid_*_history.json"))
    ncols = n + (1 if centroid_hist else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(3.8 * ncols, 4), squeeze=False)
    for i in range(1, n + 1):
        ax = axes[0][i - 1]
        lcurve = Path(f"{i}/lcurve.out")
        mace_results = sorted(glob.glob(f"{i}/results/*.txt"))
        if lcurve.is_file():
            plot_deepmd(ax, lcurve)
            ax.set_ylabel("loss / RMSE")
            ax.set_xscale("symlog")
            ax.set_yscale("log")
            ax.legend()
        elif mace_results:
            plot_mace(ax, mace_results[0], f"{i}")
        else:
            ax.set_title(f"NNP {i}: no loss file")
            continue
        ax.set_title(f"NNP {i}")
        ax.grid()

    if centroid_hist:
        ax = axes[0][n]
        plot_centroid(ax, centroid_hist[0])
        ax.set_title("centroid model")
        ax.set_ylabel(r"RMSE ($\AA$)")
        ax.set_xlim(left=0)
        ax.set_yscale("log")
        ax.grid()

    plt.tight_layout()
    plt.savefig("loss.png", dpi=150)
    print("wrote loss.png")


if __name__ == "__main__":
    main()
