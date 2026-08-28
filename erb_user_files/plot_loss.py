"""Per-NNP training-loss curves for an ArcaNN training iteration.

Copied unconditionally into each NNN-training/ folder by
training/prepare.py. Run it there once the training jobs finish:

    python plot_loss.py

Handles both engines per NNP:
  * DeePMD  -> <nnp>/lcurve.out            (whitespace table, `dp train`)
  * MACE    -> <nnp>/results/*.txt         (JSON lines, mace_run_train)

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


def plot_mace(ax, results_txt):
    rows = [json.loads(line) for line in Path(results_txt).read_text().splitlines() if line.strip()]
    evals = [r for r in rows if r.get("mode") in ("eval", "opt") and "epoch" in r]
    if not evals:
        return
    epochs = [r["epoch"] for r in evals]
    for j, key in enumerate(("rmse_e_per_atom", "rmse_f", "loss")):
        ys = [r[key] for r in evals if key in r]
        if ys:
            xs = [r["epoch"] for r in evals if key in r]
            ax.plot(xs, ys, label=key, color=COLORS[j % len(COLORS)])
    ax.set_xlabel("epoch")


def main():
    n = nnp_count()
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4), squeeze=False)
    for i in range(1, n + 1):
        ax = axes[0][i - 1]
        lcurve = Path(f"{i}/lcurve.out")
        mace_results = sorted(glob.glob(f"{i}/results/*.txt"))
        if lcurve.is_file():
            plot_deepmd(ax, lcurve)
        elif mace_results:
            plot_mace(ax, mace_results[0])
        else:
            ax.set_title(f"NNP {i}: no loss file")
            continue
        ax.set_title(f"NNP {i}")
        ax.set_ylabel("loss / RMSE")
        ax.set_xscale("symlog")
        ax.set_yscale("log")
        ax.grid()
        ax.legend()

    plt.tight_layout()
    plt.savefig("loss.png", dpi=150)
    print("wrote loss.png")


if __name__ == "__main__":
    main()
