# This branch vs. upstream ArcaNN

This fork adds an optional MACE training engine alongside stock ArcaNN's
DeePMD-kit-only workflow. Everything below is only what's *different* from
upstream install/startup — see `README.md` and the
[official docs](https://arcann-chem.github.io/arcann_training/) for
everything else, which is unchanged.

## Install

No extra Python dependencies in the ArcaNN driver env itself. MACE training
runs inside the SLURM jobs ArcaNN submits, under whatever conda env
`user_files/job_mace_train_*.sh` activates — the driver env only needs
`arcann_training` + `ase`, same as upstream. Point your job scripts at a
separate env with `mace-torch`/`torch` installed.

## New config keys (`config.json` / `input.json`, under `initialization`)

| Key | Default | Notes |
|---|---|---|
| `mlip_engine` | `"deepmd"` | Set to `"mace"` to use the MACE engine. Leave alone for stock DeePMD behavior. |
| `hydrated_electron_mode` | `false` | Opt-in. Only relevant if your system has an electron pseudo-particle; adds a companion "centroid" model trained alongside the force model. |

## If `mlip_engine: "mace"`, you additionally need in `user_files/`

- `mace_train.yaml` — MACE `run_train` config template.
- `job_mace_train_<arch_type>_<machine>.sh` — training job template.
- `job_mace_centroid_<arch_type>_<machine>.sh` — only if `hydrated_electron_mode: true`.

`erb_user_files/` has worked examples of all of these (plus `machine.json`,
labeling job scripts, etc.) to copy from — note they're KU HPC-specific
(partition names, project ID), so adjust for your own cluster/allocation.

## MACE training data

`labeling extract` (with `mlip_engine: "mace"`, CP2K labeling) writes each
iteration's labeled configs as extended XYZ, in the same format as the
initial datasets, to `data/<system>_<iter>/train.xyz` (all of them: the
validation set stays the initial datasets' `val.xyz`; disturbed candidates go
to `data/<system>-disturbed_<iter>/`). With
`hydrated_electron_mode: true`, every frame also gets the electron as a dummy
`X` atom (zero force) at the periodic centroid of the stage-2
`2_labeling_<NNNNN>-<stride>-SPIN_DENSITY-1_0.cube`, computed by
`user_files/centroid_label_from_cube.py` (+ `analyze_dataset.py`); the
centroids are also written to `data/<system>_<iter>/centroid_labels.csv`.

`training prepare` then builds `<iter>-training/{train,valid}.xyz` by
concatenating the initial datasets and every `data/<system>_<NNN>/` up to
the current iteration, and copies them to `<iter>-training/centroid/` for the
centroid model (which reads the `X` position as its target). A
`train.xyz`/`valid.xyz` (or `centroid/*.xyz`) already in place is used as is.

## Skipping labeled configs

Besides an empty `skip` file in a step directory, you can list step indices
in `<iter>-labeling/configs_to_skip.txt` (space- or newline-separated `NNNNN`,
`#` comments allowed; applies to every system). Run `labeling check` again
after changing either, then `labeling extract`.

## One small fork-specific convenience

Anything dropped in `user_files/labeling_codes/` is copied into each
labeling step's working directory automatically, before labeling runs.
Not an upstream ArcaNN feature.
