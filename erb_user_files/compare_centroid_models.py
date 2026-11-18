#!/usr/bin/env python3
"""Evaluate every centroid model on every labelled set of a hydrated-electron
(``he``) ArcaNN run, and plot the results.

ArcaNN ``training prepare`` (MACE, ``hydrated_electron_mode``) copies this file
from ``user_files/`` into ``<NNN>-training/``. ``training freeze`` then submits
it as a GPU job (``job_mace_centroid_compare_<arch>_<machine>.sh``). By hand,
from ``<NNN>-training/``, in the MACE env:

    python compare_centroid_models.py            # models 000..NNN, sets 001..NNN

How the run is laid out
-----------------------
* ``NNP/centroid_<k>.model`` is the centroid model trained in iteration *k*.
* ``data/<system>_<j>/{train,val}.xyz`` holds the configs labelled in
  ``<j>-labeling``, with the DFT spin-density centroid stored as a trailing
  ``X`` atom. Set *j* first enters training in iteration *j*.

So model *k* has **seen** set *j* when j <= k: it trained on the ``train`` part
and picked its checkpoint on the ``val`` part. When j > k the set is
**unseen**, a true held-out test. The most informative cell is model j-1 on
set j. Those are the configs exploration flagged, judged by the model that
was in use while exploring.

The error is the minimum-image distance between predicted and DFT centroid,
the same convention as the training loss.

Outputs go to ``<NNN>-training/centroid_compare/``:
    centroid_matrix.csv      one row per (model, set, split): n, RMSE, MAE,
                             median, p90, max, seen/unseen
    centroid_per_config.csv  one row per (model, config): error in A
    centroid_compare.png     the figure (also .pdf)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# These ten colours, in this order, are the dataviz skill's validated
# categorical palette (light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def find_models(work: Path, max_iter: int | None):
    out = {}
    for p in (work / "NNP").glob("centroid_[0-9][0-9][0-9].model"):
        k = int(p.stem.split("_")[1])
        if max_iter is None or k <= max_iter:
            out[k] = p
    return dict(sorted(out.items()))


def find_sets(work: Path, system: str, max_iter: int | None):
    out = {}
    pat = re.compile(rf"^{re.escape(system)}_(\d{{3}})$")
    for p in (work / "data").iterdir():
        m = pat.match(p.name)
        if m and p.is_dir():
            j = int(m.group(1))
            if j >= 1 and (max_iter is None or j <= max_iter):
                out[j] = p
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def load_frames(set_dir: Path):
    import ase.io
    from train_centroid import CENTROID_KEY, to_centroid_frames

    frames, split = [], []
    for fname, tag in (("train.xyz", "train"), ("val.xyz", "val")):
        f = set_dir / fname
        if f.is_file():
            got = to_centroid_frames(ase.io.read(f, index=":"))
            frames += got
            split += [tag] * len(got)
    true = np.array([a.info[CENTROID_KEY] for a in frames], float).reshape(-1, 3)
    cells = np.array([np.diagonal(a.cell.array) for a in frames]).reshape(-1, 3)
    configs = [str(a.info.get("config", i)) for i, a in enumerate(frames)]
    return frames, np.array(split), true, cells, configs


def predict(model_path: Path, frames, device, batch_size):
    import torch
    from mace import data
    from mace.tools import torch_geometric, torch_tools, utils

    model = torch.load(model_path, map_location=device, weights_only=False).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    # build the graphs in the model's own dtype (float32 for these checkpoints)
    torch_tools.set_default_dtype(str(next(model.parameters()).dtype).split(".")[-1])
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    dataset = [data.AtomicData.from_config(data.config_from_atoms(a, head_name="Default"),
                                           z_table=z_table, cutoff=float(model.r_max), heads=None)
               for a in frames]
    loader = torch_geometric.dataloader.DataLoader(dataset=dataset, batch_size=batch_size,
                                                   shuffle=False, drop_last=False)
    out = []
    with torch.no_grad():
        for batch in loader:
            out.append(torch_tools.to_numpy(model(batch.to(device).to_dict())["centroid"]))
    pred = np.concatenate(out).astype(float)
    assert pred.shape == (len(frames), 3)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred


def periodic_error(pred, true, cells):
    d = pred - true
    d -= cells * np.round(d / cells)
    return np.linalg.norm(d, axis=1)


def stats(err):
    return {"n": len(err), "rmse": float(np.sqrt((err ** 2).mean())), "mae": float(err.mean()),
            "median": float(np.median(err)), "p90": float(np.percentile(err, 90)),
            "max": float(err.max())}


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def make_figure(models, sets, rmse, errors, split_of, out_png: Path, system: str):
    """rmse[k][j] = RMSE of model k on all of set j; errors[(k, j)] = per-config array."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    ks, js = list(models), list(sets)
    last_j = js[-1]
    has_scatter = (last_j - 1) in models and last_j in models

    fig = plt.figure(figsize=(13.5, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3 if has_scatter else 2,
                          width_ratios=[1.15, 1.1, 0.95] if has_scatter else [1.15, 1.1])

    # (a) model x set RMSE matrix ------------------------------------------- #
    ax = fig.add_subplot(gs[0, 0])
    mat = np.array([[rmse[k][j] for j in js] for k in ks])
    blues = LinearSegmentedColormap.from_list(
        "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    im = ax.imshow(mat, cmap=blues, aspect="auto", origin="upper")
    fs = max(5.5, 9 - 0.25 * max(len(ks), len(js)))
    for a, k in enumerate(ks):
        for b, j in enumerate(js):
            v = mat[a, b]
            frac = (v - mat.min()) / (np.ptp(mat) or 1)
            ax.text(b, a, f"{v:.3f}", ha="center", va="center", fontsize=fs,
                    color="white" if frac > 0.45 else INK)
            if j > k:  # unseen -> dashed outline
                ax.add_patch(Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, lw=1.2,
                                       ec=SERIES[1], ls=(0, (3, 2))))
    ax.set_xticks(range(len(js)), [f"{j:03d}" for j in js])
    ax.set_yticks(range(len(ks)), [f"{k:03d}" for k in ks])
    ax.set_xlabel(f"labelled set ({system}_NNN)")
    ax.set_ylabel("centroid model (iteration)")
    ax.set_title("a  RMSE per model × labelled set (Å)")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.outline.set_visible(False)
    ax.plot([], [], ls=(0, (3, 2)), color=SERIES[1], lw=1.2, label="unseen by that model")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=8)

    # (b) RMSE vs model iteration, one line per set ------------------------- #
    ax = fig.add_subplot(gs[0, 1])
    for i, j in enumerate(js):
        c = SERIES[i % len(SERIES)] if len(js) <= len(SERIES) else blues(0.25 + 0.75 * i / (len(js) - 1))
        y = [rmse[k][j] for k in ks]
        ax.plot(ks, y, color=c, lw=2, zorder=2, label=f"set {j:03d}")
        seen = [k >= j for k in ks]
        ax.scatter([k for k, s in zip(ks, seen) if not s], [v for v, s in zip(y, seen) if not s],
                   s=42, facecolor=SURFACE, edgecolor=c, lw=2, zorder=3)
        ax.scatter([k for k, s in zip(ks, seen) if s], [v for v, s in zip(y, seen) if s],
                   s=42, color=c, edgecolor=SURFACE, lw=1.5, zorder=3)
        if len(js) <= 4:  # direct label at the line end
            ax.annotate(f"{j:03d}", (ks[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
    ax.set_xticks(ks, [f"{k:03d}" for k in ks])
    ax.set_xlabel("centroid model (iteration)")
    ax.set_ylabel("RMSE (Å)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("b  Error on each labelled set as models are retrained")
    ax.scatter([], [], s=42, facecolor=SURFACE, edgecolor=MUTED, lw=2, label="unseen")
    ax.scatter([], [], s=42, color=MUTED, label="seen in training")
    ax.legend(frameon=False, fontsize=8, ncol=2 if len(js) > 4 else 1,
              loc="upper center", bbox_to_anchor=(0.5, -0.16))

    # (c) per-config before/after on the newest set ------------------------- #
    if has_scatter:
        ax = fig.add_subplot(gs[0, 2])
        old, new = errors[(last_j - 1, last_j)], errors[(last_j, last_j)]
        sp = split_of[last_j]
        lim = 1.05 * max(old.max(), new.max())
        ax.fill_between([0, lim], [0, lim], [0, 0], color=SERIES[2], alpha=0.07, lw=0)
        ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--", zorder=1)
        for tag, c, mk in (("train", SERIES[0], "o"), ("val", SERIES[1], "D")):
            m = sp == tag
            if m.any():
                ax.scatter(old[m], new[m], s=26 if mk == "o" else 30, marker=mk, color=c,
                           edgecolor=SURFACE, lw=0.8, zorder=3, label=f"{tag} split (n={m.sum()})")
        better = int((new < old).sum())
        ax.text(0.97 * lim, 0.04 * lim, f"below diagonal = improved\n{better}/{len(old)} configs", ha="right", va="bottom",
                fontsize=8, color=INK2)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"centroid_{last_j - 1:03d} error (Å)  — before")
        ax.set_ylabel(f"centroid_{last_j:03d} error (Å)  — after")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"c  Set {last_j:03d}, per config")
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.savefig(out_png, dpi=200)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", type=Path, default=HERE.parent,
                    help="ArcaNN work dir holding NNP/, data/, user_files/ (default: ..)")
    ap.add_argument("--max-iter", type=int, default=None,
                    help="last iteration to include (default: this <NNN>-training's, else all)")
    ap.add_argument("--system", default="he", help="labelled-set prefix in data/ (default he)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: ./centroid_compare)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figure from an existing centroid_per_config.csv")
    args = ap.parse_args(argv)

    work = args.work_dir.resolve()
    sys.path.insert(0, str(work / "user_files"))  # model.py / loss.py / train_centroid.py
    max_iter = args.max_iter
    if max_iter is None and re.match(r"^\d{3}-training$", HERE.name):
        max_iter = int(HERE.name[:3])
    out = (args.out or HERE / "centroid_compare").resolve()
    out.mkdir(parents=True, exist_ok=True)

    models = find_models(work, max_iter)
    sets = find_sets(work, args.system, max_iter)
    if not models:
        sys.exit(f"no NNP/centroid_NNN.model under {work}")
    if not sets:
        print(f"no data/{args.system}_NNN labelled sets (iteration 0?) -- nothing to compare")
        return 0
    print(f"work dir: {work}")
    print(f"models  : {', '.join(f'{k:03d}' for k in models)}")
    print(f"sets    : {', '.join(f'{args.system}_{j:03d}' for j in sets)}")

    errors, split_of, configs_of = {}, {}, {}
    per_config_csv = out / "centroid_per_config.csv"
    if args.plot_only:
        with per_config_csv.open() as fh:
            rows = list(csv.DictReader(fh))
        for j in sets:
            r_j = [r for r in rows if int(r["set"]) == j]
            first = [r for r in r_j if int(r["model"]) == min(models)]
            split_of[j] = np.array([r["split"] for r in first])
            configs_of[j] = [r["config"] for r in first]
            for k in models:
                errors[(k, j)] = np.array([float(r["err_A"]) for r in r_j if int(r["model"]) == k])
    else:
        import mace  # noqa: F401  -- import before the user_files modules
        import torch
        from mace.tools import torch_tools

        device = torch_tools.init_device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"device  : {device}\n")
        # all sets concatenated -> one inference pass per model
        loaded = {j: load_frames(p) for j, p in sets.items()}
        all_frames = [f for j in sets for f in loaded[j][0]]
        bounds, start = {}, 0
        for j in sets:
            n = len(loaded[j][0])
            bounds[j] = slice(start, start + n)
            start += n
            split_of[j], configs_of[j] = loaded[j][1], loaded[j][4]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # cuequivariance / ase chatter
            for k, path in models.items():
                pred = predict(path, all_frames, device, args.batch_size)
                for j in sets:
                    _, _, true, cells, _ = loaded[j]
                    errors[(k, j)] = periodic_error(pred[bounds[j]], true, cells)
                print(f"  evaluated {path.name}")
        with per_config_csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "set", "config", "split", "seen", "err_A"])
            for (k, j), e in errors.items():
                for c, s, v in zip(configs_of[j], split_of[j], e):
                    w.writerow([f"{k:03d}", f"{j:03d}", c, s, int(j <= k), f"{v:.6f}"])

    # matrix CSV + console table
    rmse = {k: {} for k in models}
    with (out / "centroid_matrix.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "set", "split", "seen", "n", "rmse_A", "mae_A", "median_A", "p90_A", "max_A"])
        for k in models:
            for j in sets:
                e = errors[(k, j)]
                for tag, m in (("all", np.ones(len(e), bool)), ("train", split_of[j] == "train"),
                               ("val", split_of[j] == "val")):
                    if not m.any():
                        continue
                    s = stats(e[m])
                    if tag == "all":
                        rmse[k][j] = s["rmse"]
                    w.writerow([f"{k:03d}", f"{j:03d}", tag, int(j <= k), s["n"]]
                               + [f"{s[x]:.6f}" for x in ("rmse", "mae", "median", "p90", "max")])

    print("\nRMSE (Å) on all configs of each set; * = set unseen by that model")
    print("model  " + "".join(f"{'set ' + format(j, '03d'):>12}" for j in sets))
    for k in models:
        print(f"{k:03d}    " + "".join(f"{rmse[k][j]:>11.4f}{'*' if j > k else ' '}" for j in sets))

    print("\nEach set: model before it was labelled (k=j-1, unseen) -> model trained on it (k=j)")
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    for j in sets:
        if (j - 1) not in models or j not in models:
            continue
        old, new = errors[(j - 1, j)], errors[(j, j)]
        line = (f"  set {j:03d} (n={len(old)}): RMSE {rmse[j - 1][j]:.4f} -> {rmse[j][j]:.4f} A "
                f"(x{rmse[j][j] / rmse[j - 1][j]:.2f}); improved {(new < old).sum()}/{len(old)}")
        if wilcoxon is not None and np.any(new != old):
            line += f"; Wilcoxon p={wilcoxon(new, old, alternative='less').pvalue:.1e}"
        last = max(models)
        if last > j:
            line += f"; latest {last:03d}: {rmse[last][j]:.4f} A"
        print(line)

    png = out / "centroid_compare.png"
    make_figure(models, sets, rmse, errors, split_of, png, args.system)
    print(f"\noutputs -> {out}  (centroid_matrix.csv, centroid_per_config.csv, {png.name}/.pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
