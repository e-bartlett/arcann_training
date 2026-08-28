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

Commit: <hash>

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
