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

Commit: <hash>
