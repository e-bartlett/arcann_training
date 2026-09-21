#!/usr/bin/env python3
"""MACE tandem MD driver -- ArcaNN exploration step (``<system>_mace_lammps.py``).

Thin wrapper: ArcaNN's ``exploration/prepare.py`` reads this as
``user_files/<system>_mace_lammps.py`` for the mace engine (the DeePMD path
keeps its ``_naive_lammps.py`` name), fills the ``_R_`` slots per trajectory,
the explore job puts ``dataset_prep/mace_electron/`` on ``PYTHONPATH``, and
the real driver body lives in ``he_mace_md_mliap.py`` there.

Default exploration path (MD_PERFORMANCE_PLAN.md Phase 3.5, ~0.72 ns/day on
A40, ~5x over the e3nn ``pair_style mace`` driver): the cuEquivariance /
LAMMPS ML-IAP route. Model 0 in ``_R_MODEL_FILES_`` is the
``mace_<nnp>_<iter>.model-mliap_lammps.pt`` pickle
(``mace_create_lammps_model --format=mliap``); members 1..K-1 are the plain
``.model`` siblings for the Python committee eval. Needs the develop-branch
ML-IAP+Kokkos LAMMPS build (``dataset_prep/mace_electron/build_lammps_mliap.sh``,
Kokkos_ARCH_AMPERE86) in the ``mace_electron_cueq`` env, run under
``mpirun -n 1`` with ``-k on g 1 -sf kk`` (set by the driver).

To fall back to the e3nn ``pair_style mace`` driver, change the import to
``from he_mace_md import run`` and point ``_R_MODEL_FILES_`` model 0 at the
``.model-lammps.pt`` (libtorch) file instead.
"""

from he_mace_md_mliap import run

run(
    lammps_in="_R_LAMMPS_IN_",
    n_steps=_R_NUMBER_OF_STEPS_,
    print_freq=_R_PRINT_FREQ_,
    model_files="_R_MODEL_FILES_".split(),
    centroid_model="_R_CENTROID_MODEL_",
    devi_out="_R_DEVI_OUT_",
    restart_out="_R_RESTART_OUT_",
    vel_seed="_R_SEED_VEL_",
    temperature=_R_TEMPERATURE_,
)
