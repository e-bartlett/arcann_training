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
