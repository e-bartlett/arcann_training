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

## Biggest behavioral difference: no automated MACE data prep

ArcaNN does **not** generate MACE training data. Before running
`training prepare` on a MACE iteration, you must already have:

- `<iter>-training/train.xyz` and `valid.xyz`
- `<iter>-training/centroid/train.xyz` — only if `hydrated_electron_mode: true`

prepared by you (or your own scripts), outside of ArcaNN. `training prepare`
just checks these exist, stages the per-NNP job files, and (with
`hydrated_electron_mode`) the centroid job — it errors clearly, naming the
missing file, if they aren't there yet.

## One small fork-specific convenience

Anything dropped in `user_files/labeling_codes/` is copied into each
labeling step's working directory automatically, before labeling runs.
Not an upstream ArcaNN feature.
