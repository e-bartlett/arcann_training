# Code review queue — `mace-tandem` branch

This fork branch adopts the hydrated_electron repo's review convention
(see that repo's `CLAUDE.md`). Every turn that touches `.py` / `.sh` /
templates on this branch appends one dated entry here, newest first, with a
`[ ]` per file and the commit hash filled in after committing. The same
entry is mirrored in the hydrated_electron repo's `CODE_REVIEW_QUEUE.md`
with these paths prefixed `arcann_training fork ·`.

Do not check boxes yourself — that is the reviewer marking a file done.

Chunk plan and per-chunk gates: `ARCANN_TANDEM_PLAN.md` in the
hydrated_electron repo ("Review-queue staging" section).

---

## 2026-08-28 — Chunk 3 follow-up 3: hard-code the driver name to `_mace_lammps.py`

Per the user: drop the `mlip_engine` conditional added in follow-up 2 — this
branch drives LAMMPS from the MACE tandem driver unconditionally.

- [ ] `arcann_training/exploration/prepare.py` — `driver_script_suffix` is
  now the plain constant `"_mace_lammps.py"` (was
  `"_mace_lammps.py" if mlip_engine == "mace" else "_naive_lammps.py"`).
  The four use-sites are unchanged. Consequence: on `mace-tandem` a DeePMD
  `exploration prepare` would also look for `<system>_mace_lammps.py` — fine,
  the branch is MACE-only. Full unit suite 166/167 (same pre-existing
  unrelated `test_check.py` error).

Commit: 441baa7

---

## 2026-08-28 — Chunk 3 follow-up 2: MACE driver script name `he_mace_lammps.py`

User asked to rename the repo's `erb_user_files/he_naive_lammps.py` →
`he_mace_lammps.py`. ArcaNN hard-codes the `<system>_naive_lammps.py`
filename in `exploration/prepare.py`, so the fork needs to look for the new
name under the mace engine.

- [ ] `arcann_training/exploration/prepare.py` — in the `lammps` block, new
  `driver_script_suffix = "_mace_lammps.py" if main_json.get("mlip_engine",
  "deepmd") == "mace" else "_naive_lammps.py"`, used in all four places the
  old literal `_naive_lammps.py` appeared: reading
  `user_files/<system>_naive_lammps.py`, writing
  `<system>_<nnp>_<iter>_naive_lammps.py` into the run dir, the job-array
  `.lst` line, and the `_R_LAMMPS_PYTHON_SCRIPT_` job-file substitution.
  DeePMD path unchanged (suffix resolves to `_naive_lammps.py`). Full unit
  suite 166/167 (same pre-existing unrelated `test_check.py` error).

Commit: c761e33

---

## 2026-08-28 — Chunk 3 follow-up: also symlink the plain `.model` committee files

Surfaced writing the Chunk 4 driver (`dataset_prep/mace_electron/he_mace_md.py`
in the hydrated_electron repo). The Python committee force eval needs the
plain `mace_<n>_<iter>.model` checkpoints -- the LAMMPS-compiled
`.model-lammps.pt` (what `pair_style mace` loads) can't be loaded by a
`MACECalculator`-style path. `create_models_list`'s mace branch previously
symlinked only the `.model-lammps.pt`.

- [ ] `arcann_training/exploration/utils.py` — `create_models_list` mace
  branch now symlinks **both** `mace_<n>_<prev_iter>.model-lammps.pt` and
  `mace_<n>_<prev_iter>.model` into `local_path` per NNP. `models_list` /
  `models_string` (and thus `_R_MODEL_FILES_`, `pair_coeff`) are unchanged
  -- still the `.model-lammps.pt` names only. `freeze.py` already produces
  both files in `NNP/`.
- [ ] `arcann_training/unittests/test_utils_exploration.py` —
  `TestCreateModelsListMace` now also `touch`es the `.model` siblings in
  the fake `NNP/` and asserts both symlinks per NNP resolve into `NNP/`.
  Suite still 4/4 for this module (166/167 overall, same pre-existing
  unrelated `test_check.py` error).

Commit: a57e619

---

## 2026-08-28 — Chunk 3: exploration spot fixes (utils + prepare)

Phase-1 exploration spot fixes from `ARCANN_TANDEM_PLAN.md`. Same gate as
Chunks 1-2: `main_json.get("mlip_engine", "deepmd") == "mace"`; the DeePMD
fall-through is byte-identical (both mace branches are a leading `if ...:`
that either early-returns or adds a dict key, nothing in the existing path
moved). `deviate.py` deliberately untouched — the plan defers that to P3
(only add a 3-line `model_deviation = raw` branch if the driver's
`model_devi_*.out` does not survive the EB decimation slice).

- [ ] `arcann_training/exploration/utils.py` — `create_models_list`: mace
      branch before the compressed-model logic. Builds
      `mace_<nnp>_<prev_iter>.model-lammps.pt` names (committee reorder
      unchanged: propagated NNP first), symlinks each
      `NNP/mace_<n>_<prev_iter>.model-lammps.pt` into `local_path`, returns
      the same `(models_list, models_string)` tuple, then `return`s so the
      DeePMD body never runs. Does **not** read `previous_json["is_compressed"]`
      (no compression for MACE).
- [ ] `arcann_training/exploration/prepare.py` — two mace branches in
      `main()`:
      1. `deepmd_model_version` seed: `previous_training_json["deepmd_
         model_version"]` was a hard index (KeyErrors on a hand-seeded
         `training_000.json` with no such key). Now: mace →
         `.get("deepmd_model_version", "mace")`; deepmd → the original hard
         index, unchanged.
      2. LAMMPS branch, inside the per-NNP/per-traj loop, right after
         `_R_ITER_`: for mace, symlink `NNP/centroid_<prev_iter>.model` into
         `local_path` and set `input_replace_dict["_R_CENTROID_MODEL_"]` to
         its **basename** (`centroid_<prev_iter>.model`). Basename (not the
         plan's literal `NNP/centroid_<prev_iter>.model` string) so it
         matches how `create_models_list`'s committee symlinks are consumed:
         the explore job `realpath`s each basename from `local_path` and
         links it into its scratch workdir. Flag if the plan intended a
         literal path here instead.
- [ ] `arcann_training/unittests/test_utils_exploration.py` — new
      `TestCreateModelsListMace` (own temp-dir setUp/tearDown, does not
      touch `TestCreateModelsList`). Asserts the mace filenames, the
      space-joined string, and that all `nnp_count` symlinks resolve into
      `NNP/`, with `is_compressed` absent from the prev-training JSON.
      Full suite: `python -m unittest discover -s arcann_training/unittests`
      → 166/167 (the 1 error is the pre-existing unrelated
      `test_check.py::test_validate_step_folder`).

Gate before running `exploration prepare` (per the plan's staging table).

Commit: 44de52ba

---

## 2026-08-28 — Chunk 2 (part B): training/prepare.py MACE branch

Finishes Chunk 2. Early-return design (user's call): `main()` gains one
`if main_json.get("mlip_engine","deepmd") == "mace": return _prepare_mace(...)`
right after the labeling-extracted gate, so the entire DeePMD body
(dptrain discovery, `validate_deepmd_config`, `dp_train_input`, the exp-LR
recompute, the per-NNP `training.json`) is untouched.

- [ ] `arcann_training/training/prepare.py` — new module fn `_prepare_mace`
      (~200 lines) + the 1-line early return in `main`. What it does:
  - `generate_training_json` for the shared user>previous>default merge
    (popping any string `deepmd_model_version` first — it type-checks
    against the numeric default), then forces `deepmd_model_version="mace"`.
  - discovers `user_files/mace_train_r<N>.yaml` (highest N) and
    `job_mace_train_<arch>_<machine>.sh`.
  - collects data dirs: `data/init_*` (from `check_initial_datasets`) +
    `data/<sys>_<iter>` for iter 1..curr, sys in `systems_auto`. For a
    192-particle dir it pairs `<iter>-exploration/<sys>/elec_candidates_
    <iter>_<sys>.xyz`; a 193-particle dir already has X.
    **Assumption to verify against a live work dir**: that candidates path,
    and that `data/<sys>_<iter>` frame order == elec_candidates line order.
    -disturbed / adhoc / extra_ dirs are not handled (systems_auto:["he"]
    only) — noted in-code.
  - epochs/walltime (**P2 tunes**): `numb_steps` <= 10000 is read as an
    explicit epoch count, else default 400; `mean_s_per_step` default
    150 s/epoch (~1.5x the L40 survey figure); walltime = ceil(epochs *
    s/epoch * 1.3) rounded to the hour.
  - sets `is_prepared=True`, `is_compress_launched=is_compressed=True`
    (compression n/a), the rest False.
  - runs `<mace_env>/bin/python user_files/deepmd_npy_to_extxyz.py --out
    <iter>-training --seed <padded_iter> --deepmd-dir ... [--elec-xyz ...]`
    once → `<iter>-training/{train,valid}.xyz`, rsynced into each NNP.
  - per NNP: `mace_train.yaml` from the template with `_R_SEED_` /
    `_R_MACE_MAX_EPOCHS_` filled; `job_mace_train_<arch>_<machine>.sh` via
    `replace_in_slurm_file_general` + `_R_MACE_CONFIG_` / `_R_MACE_NAME_`
    (= `mace_<nnp>_<iter>`) / `_R_MACE_LOG_` (= `training.log`) / `_R_SEED_`.
  - dumps `config.json`, `training_<iter>.json`, `used_input.json`.
  `prepare.py` parses + imports; 165/166 unit tests pass (the 1 failure is
  the pre-existing `test_check.py::test_validate_step_folder`).
- [ ] `erb_user_files/deepmd_npy_to_extxyz.py` — MIRROR of the tested
      `dataset_prep/mace_electron/` copy (header points there); this is what
      ArcaNN init drops in `$WORK_DIR/user_files/` for `_prepare_mace`.

Commit: 5a0eff5

---

## 2026-08-28 — Chunk 2 (part A): MACE train user-files (converter + yaml + job + plot)

The standalone repo/user-file pieces of Chunk 2. `training/prepare.py`'s
MACE branch (part B) wires them in next — held for a design check.

- [ ] `erb_user_files/mace_train_r0.yaml` — `mace_run_train --config`
      template (architecture from the plan: MACE, r_max 6, `128x0e+128x1o`,
      2 interactions, correlation 3, max_ell 3; weighted force loss +
      stage-two; SWA/EMA; `default_dtype: float32`). `E0s` keyed by Z:
      `{0: 0.0, 1: 9.600156, 8: -6.505675}` (O/H from `dptrain_3.0.json`
      `atom_ener`, X = 0). `_R_` slots `_R_SEED_`, `_R_MACE_MAX_EPOCHS_`
      (prepare.py fills); name/seed/files stay on the CLI. **P2 checks: does
      `mace_run_train` accept Z 0 in the z-table? does the O/H E0s scale
      match?** — noted in-file. Verified: YAML parses; post-substitution
      the two slots become ints.
- [ ] `erb_user_files/job_mace_train_gpu_login1.sh` — ArcaNN job template
      (`job_mace_train_<arch>_<machine>.sh`), mirrors
      `job_deepmd_train_gpu_login1.sh`'s header (`_R_PARTITION_` /
      `_R_SUBPARTITION_`). `_R_` slots: `_R_MACE_CONFIG_` (=mace_train.yaml),
      `_R_MACE_NAME_` (=mace_<nnp>_<iter>), `_R_MACE_LOG_` (=training.log —
      the Chunk-1 contract), `_R_SEED_`. Activates `mace_electron`, runs
      `mace_run_train --model_dir . --log_dir . --results_dir ./results
      --train_file train.xyz --valid_file valid.xyz > training.log`; no
      `../data` copy (the converter writes the xyz into `<nnp>/`). Backgrounds
      an `nvidia-smi` poller → `gpu_poll.csv`. `bash -n` clean. NOTE: the
      machine keyword's sub-partition must pin `nvidia&(l40|a40|a100)`.
- [ ] `erb_user_files/plot_loss.py` — new (did not exist; the fork's
      `prepare.py` line ~110 already copies `user_files/plot_loss.py`
      unconditionally, so DeePMD needed it too). Per-NNP: `<nnp>/lcurve.out`
      → DeePMD table plot; else `<nnp>/results/*.txt` → MACE JSON-lines
      plot (`rmse_e_per_atom` / `rmse_f` / `loss` vs epoch). `nnp_count`
      from `../control/config.json`. Style follows the sibling
      `erb_user_files/plot/plot_lcurve.py`, not the repo analysis-plot
      convention. Smoke-tested on a synthetic MACE results dir.

The same three files are added to the repo's `erb_user_files/` (mirror) and
`dataset_prep/mace_electron/deepmd_npy_to_extxyz.py` lives repo-side only —
see the repo `CODE_REVIEW_QUEUE.md`.

Commit: f523b78

---

## 2026-08-28 — Chunk 1 (rest): training freeze/check/increment + the init gate

The remaining Phase-1 training spot fixes plus the initialization-side gate
the plan's file list omitted. Same gate everywhere:
`main_json.get("mlip_engine", "deepmd") == "mace"` (a leading `if`; the
`else`/fall-through DeePMD path is byte-identical). `_R_MACE_NAME_ =
mace_<nnp>_<iter>` is now locked (user, 2026-08-28), so every consumer keys
on `<nnp>/mace_<nnp>_<iter>.model` and `NNP/mace_<nnp>_<iter>.model-lammps.pt`.

`training/prepare.py` is **not** here — moved to Chunk 2 (its MACE branch
calls the Chunk-2 converter and stages the Chunk-2 yaml/job templates).

- [ ] `arcann_training/assets/default_config.json` — `initialization` gains
      `"mlip_engine": "deepmd"` and `"mace_env": ""`. `generate_main_json`
      type-checks against these (both str) and stamps them into every
      `config.json`; DeePMD reads `mlip_engine` as `"deepmd"` → unchanged.
      The `generate_main_json` unit tests use their own inline default dict,
      so they're unaffected (verified: 165/165 relevant tests pass; the one
      failure, `test_check.py::test_validate_step_folder`, is pre-existing —
      `with <Path>:` in the test, fails identically on the clean tree).
- [ ] `arcann_training/initialization/start.py` — skip
      `check_dptrain_properties()` when `mlip_engine == "mace"` (it raises
      `FileNotFoundError` with no `dptrain_*.json`).
- [ ] `arcann_training/training/launch.py` — job-file stem
      `job_mace_train_…` vs `job_deepmd_train_…` via one `train_job_stem`
      var; loop now builds the name once and reuses it.
- [ ] `arcann_training/training/check.py` — MACE branch (returns before the
      DeePMD body): done ⟺ every `<nnp>/mace_<nnp>_<iter>.model` exists and
      `<nnp>/training.log` tail contains a lone `"Done"` (mace_run_train's
      last line). Sets `is_checked`. No ckpt rename, no nbor parse.
      `mean_s_per_step` left as `prepare.py` set it — observed epoch timing
      is in MACE's `results/*.txt`, not `training.log`; parsing it is a
      later refinement (noted in-code).
- [ ] `arcann_training/training/check_freeze.py` — MACE branch: per-NNP
      success is `NNP/mace_<nnp>_<iter>.model-lammps.pt` existing (not a
      per-folder `graph_*.pb`); sets `is_frozen` when all present.
- [ ] `arcann_training/training/freeze.py` — MACE branch (returns before the
      machine-spec/sbatch code): no job. Runs
      `<mace_env>/bin/mace_create_lammps_model --dtype float32
      <nnp>/mace_<nnp>_<iter>.model` (falls back to PATH if `mace_env`
      unset), rsyncs the `.model` + produced `.model-lammps.pt` into
      `NNP/`, sets `is_freeze_launched`/`is_frozen`. float32 = LAMMPS
      pair_style mace's dtype (plan GPU section). Single-head models don't
      prompt; a multi-head model would need `--head` (not expected here).
- [ ] `arcann_training/training/increment.py` — MACE branch: `check_file_
      existence` on `<nnp>/mace_<nnp>_<iter>.model`; skip the `is_compressed`
      / `_compressed.pb` checks and the `graph_*.pb → NNP/` rsync (freeze
      already populated `NNP/`). Test-folder + data rsync + next-iter
      folders + `current_iteration` bump unchanged.

Cross-chunk contract for Chunk 2's `job_mace_train_*.sh` / `mace_train_*.yaml`:
`mace_run_train --name mace_<nnp>_<iter> --model_dir <nnp>` and redirect
stdout to `<nnp>/training.log`.

Commit: d532c5c

---

## 2026-08-28 — Chunk 1 (partial): training spot fixes — no-op / guard subset

Phase 1 of the plan, the three files with a self-contained MACE branch (no
dependency on Chunk-2 naming). Gate on the optional key
`main_json["mlip_engine"] == "mace"` (absent → `"deepmd"`, every existing
path byte-identical). The remaining six training files (`prepare`, `launch`,
`check`, `freeze`, `check_freeze`, `increment`) are held pending a design
check with the user — they interlock on the `<name>` / job-filename / YAML
conventions defined in Chunk 2, and there is an init-side gap
(`initialization/start.py` → `check_dptrain_properties`) the plan's file
list doesn't cover.

- [ ] `arcann_training/training/utils.py` — `validate_deepmd_config`: early
      `return` when `training_config["deepmd_model_version"] == "mace"`
      (the value `training/prepare.py` will set in its MACE branch), before
      the `float(...)` version check that would otherwise raise.
- [ ] `arcann_training/training/compress.py` — `main`: right after the JSONs
      load, if `mlip_engine == "mace"` log "n/a", set
      `is_compress_launched = is_compressed = True`, write the training JSON,
      `return 0`. No machine spec, no job file, no sbatch.
- [ ] `arcann_training/training/check_compress.py` — `main`: same short-circuit
      (`is_compressed = True`, `return 0`).

DeePMD path is untouched in all three (branch is a leading `if`, `else`
falls through to the original body). `ast.parse` + import-check pass under
the `arcann` env's editable install.

Commit: f5a825f

---

## 2026-08-28 — Chunk 0: restructure fork into an editable `arcann_training` package

Phase 0a of the plan. No behavioural change to any ArcaNN code path — this
is purely a layout move so the `arcann` conda env can `pip install -e` the
fork and pick up subsequent `mace-tandem` edits live.

- [ ] `git mv erb_arcann_training → arcann_training` — the hydrated-electron
      modified package replaces the stock upstream copy. The 12 files that
      differ from upstream show as `M` on the new path; the rest as pure
      renames. Content is byte-identical to what the `arcann` env was
      already running (verified against a pre-change snapshot).
- [ ] `.gitignore` — added `__pycache__/` and `*.pyc` (bytecode was
      previously tracked in `erb_arcann_training/`; dropped from the index
      in this move).
- [ ] `setup.py` — unchanged; `find_packages()` now resolves to
      `arcann_training.*` only (confirmed).

Editable install verified in the `arcann` env: every `arcann_training.*`
submodule resolves into this fork, `python -m arcann_training --help` and
all phase entrypoints import with no regression.

Commit: c8ac2ce
