"""Per-NNP training-loss curves for an ArcaNN training iteration.

Copied unconditionally into each NNN-training/ folder by
training/prepare.py. Run it there once the training jobs finish:

    python plot_loss.py

Handles both engines per NNP:
  * DeePMD  -> <nnp>/lcurve.out            (whitespace table, `dp train`)
  * MACE    -> <nnp>/results/*.txt         (JSON lines, mace_run_train)

Each NNP gets two stacked panels (energy on top, forces on bottom), each
showing train and valid together:
  * DeePMD: continuous train/valid RMSE straight from lcurve.out's
    rmse_{e,f}_{trn,val} columns (val columns are absent/skipped if no
    validation set was configured for that run).
  * MACE: mace_run_train only logs a per-epoch RMSE on the *valid* set
    (the JSON "eval" rows, stage one only -- see mace_error_tables below
    for why). Train-set RMSE only exists at two points: the final
    "Error-table on TRAIN and VALID" mace_run_train prints in
    <nnp>/training.log once for the stage-one model and once for the
    deployed SWA stage-two model. Those two points are plotted as
    train/valid marker pairs on top of the continuous valid curve, so the
    stage-two jump (usually 10-30x better than anything in stage one) is
    visible even though it's never part of the per-epoch curve.

If the centroid (electron-position) model has run this iteration, its
per-epoch history (centroid/centroid_<iter>_history.json, written by
train_centroid.py) gets an extra panel spanning both rows on the right --
it already reports train_rmse/valid_rmse together, so it doesn't need the
energy/force split.

Writes loss.png next to this script.
"""

import glob
import json
import re
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


def plot_deepmd(ax_e, ax_f, lcurve):
    data = np.genfromtxt(lcurve, names=True)
    names = data.dtype.names
    for ax, quantity in ((ax_e, "e"), (ax_f, "f")):
        for suffix, label, color in (("trn", "train", "k"), ("val", "valid", "b")):
            col = f"rmse_{quantity}_{suffix}"
            if col in names:
                ax.plot(data["step"], data[col], color=color, label=label)
        ax.set_xscale("symlog")
        ax.set_yscale("log")
        ax.legend()
    ax_e.set_ylabel("E RMSE")
    ax_f.set_ylabel("F RMSE")
    ax_f.set_xlabel("step")


def mace_error_tables(nnp_dir):
    """[{epoch, train_e, valid_e, train_f, valid_f}, ...] -- one entry per
    "Error-table on TRAIN and VALID" block in training.log (meV/atom,
    meV/A, in the order mace_run_train prints them: stage-one final model,
    then the deployed SWA stage-two model). Empty list if the run hasn't
    reached that point yet."""
    log = Path(nnp_dir) / "training.log"
    if not log.is_file():
        return []
    lines = log.read_text().splitlines()
    records = []
    pending_epoch = None
    for i, ln in enumerate(lines):
        m = re.search(r"Loaded Stage \w+ model from epoch (\d+) for evaluation", ln)
        if m:
            pending_epoch = int(m.group(1))
            continue
        if pending_epoch is not None and "| train_Default |" in ln and i + 1 < len(lines):
            train_cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            valid_cells = [c.strip() for c in lines[i + 1].strip().strip("|").split("|")]
            records.append({
                "epoch": pending_epoch,
                "train_e": float(train_cells[1]), "train_f": float(train_cells[2]),
                "valid_e": float(valid_cells[1]), "valid_f": float(valid_cells[2]),
            })
            pending_epoch = None
    return records


def plot_mace(ax_e, ax_f, results_txt, nnp_dir):
    rows = [json.loads(line) for line in Path(results_txt).read_text().splitlines() if line.strip()]
    evals = [r for r in rows if r.get("mode") == "eval" and r.get("epoch") is not None]
    if evals:
        epochs = [r["epoch"] for r in evals]
        ax_e.plot(epochs, [r["rmse_e_per_atom"] * 1000 for r in evals], color="r", label="valid (stage one)")
        ax_f.plot(epochs, [r["rmse_f"] * 1000 for r in evals], color="r", label="valid (stage one)")

    tables = mace_error_tables(nnp_dir)
    if tables:
        t_epochs = [t["epoch"] for t in tables]
        ax_e.plot(t_epochs, [t["train_e"] for t in tables], "o--", color="k", label="train (final)")
        ax_e.plot(t_epochs, [t["valid_e"] for t in tables], "o--", color="b", label="valid (final)")
        ax_f.plot(t_epochs, [t["train_f"] for t in tables], "o--", color="k", label="train (final)")
        ax_f.plot(t_epochs, [t["valid_f"] for t in tables], "o--", color="b", label="valid (final)")

    ax_e.set_ylabel("E RMSE (meV/atom)")
    ax_e.set_yscale("log")
    ax_e.legend(fontsize=7)

    ax_f.set_ylabel(r"F RMSE (meV/$\AA$)")
    ax_f.set_yscale("log")
    ax_f.set_xlabel("epoch")
    ax_f.legend(fontsize=7)


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
    fig = plt.figure(figsize=(3.8 * ncols, 8))
    gs = fig.add_gridspec(2, ncols)

    for i in range(1, n + 1):
        ax_e = fig.add_subplot(gs[0, i - 1])
        ax_f = fig.add_subplot(gs[1, i - 1], sharex=ax_e)
        lcurve = Path(f"{i}/lcurve.out")
        mace_results = sorted(glob.glob(f"{i}/results/*.txt"))
        if lcurve.is_file():
            plot_deepmd(ax_e, ax_f, lcurve)
        elif mace_results:
            plot_mace(ax_e, ax_f, mace_results[0], f"{i}")
        else:
            ax_e.set_title(f"NNP {i}: no loss file")
            ax_f.set_visible(False)
            continue
        ax_e.set_title(f"NNP {i}")
        ax_e.grid()
        ax_f.grid()

    if centroid_hist:
        ax = fig.add_subplot(gs[:, n])
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
