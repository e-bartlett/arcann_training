"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#

MACE branch of ``labeling extract`` (see ``extract.main``'s early return).

Instead of DeePMD ``set.000/*.npy``, every labeled (non-skipped) config of a
system is written as one extended XYZ frame in the same format as the
initial datasets (``data/init_*/{train,val}.xyz``)::

    Lattice="..." Properties=species:S:1:pos:R:3:forces_REF:R:3 energy_REF=<eV> config=<sys>_<iter>_<NNNNN> pbc="T T T"

into ``data/<system>_<iter>/train.xyz`` (disturbed candidates go to
``data/<system>-disturbed_<iter>/``). No frames are held out: the validation
set stays the initial datasets' ``val.xyz``. ``training prepare`` concatenates
these with the initial datasets into the next ``<iter>-training/train.xyz``.

With ``hydrated_electron_mode``, each frame also gets the electron as a dummy
``X`` atom (zero force) at the periodic centroid of the stage-2 CP2K spin
density cube, ``2_labeling_<NNNNN>-<stride>-SPIN_DENSITY-1_0.cube``. The
centroid is computed by the project's ``user_files/centroid_label_from_cube.py``
(so it matches the initial datasets' labels), and every row is also written to
``data/<system>_<iter>/centroid_labels.csv``. The centroid model reads the
same ``X`` position as its target, so both models train on the same frames.
"""

# Standard library modules
import csv
import importlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Non-standard library imports
import numpy as np

# Local imports
from arcann_training.common.list import textfile_to_string_list
from arcann_training.common.parsing_labeling import (
    extract_and_convert_box_volume,
    extract_and_convert_energy,
    extract_and_convert_forces,
)
from arcann_training.labeling.utils import is_config_skipped

# Symbol of the electron pseudo-particle (as in the initial datasets).
ELECTRON_SYMBOL = "X"
# How negative spin density is handled for the centroid (as in spin_analysis.csv).
CENTROID_NEGATIVE = "clip"


def _first_existing(*paths: Path) -> Path:
    """First path that exists, else the last one (so the error names it)."""
    for path in paths:
        if path.is_file():
            return path
    return paths[-1]


def _load_centroid_module(user_files_path: Path):
    """Import ``user_files/centroid_label_from_cube.py`` (and its
    ``analyze_dataset.py``) from the project."""
    if not (user_files_path / "centroid_label_from_cube.py").is_file():
        raise FileNotFoundError(
            f"hydrated_electron_mode needs {user_files_path / 'centroid_label_from_cube.py'} "
            f"(and analyze_dataset.py); copy them from erb_user_files/."
        )
    if str(user_files_path) not in sys.path:
        sys.path.insert(0, str(user_files_path))
    return importlib.import_module("centroid_label_from_cube")


def _cube_stride(centroid_module, labeling_step_path: Path) -> Optional[int]:
    """Stride of the step's SPIN_DENSITY cube; the finest if several."""
    selected, _ = centroid_module.select_grains(labeling_step_path, None)
    strides = sorted(stride for stride in selected if stride is not None)
    return strides[0] if strides else None


def _read_labeling_xyz(xyz_path: Path) -> Tuple[List[str], np.ndarray]:
    """Symbols and positions (Angstrom) of a labeling_<NNNNN>.xyz."""
    lines = textfile_to_string_list(xyz_path)
    atom_count = int(lines[0].split()[0])
    atom_lines = [line.split() for line in lines[2 : 2 + atom_count]]
    symbols = [fields[0] for fields in atom_lines]
    positions = np.asarray([fields[1:4] for fields in atom_lines], dtype=np.float64)
    return symbols, positions


def _label_step(
    labeling_step_path: Path,
    padded_labeling_step: str,
    Ha_to_eV: float,
    au_to_eV_per_A: float,
    program_version: float,
) -> Dict:
    """Parse one CP2K-labeled step: symbols, positions, forces, energy, cell, pbc."""
    symbols, positions = _read_labeling_xyz(
        _first_existing(
            labeling_step_path / f"naive_labeling_{padded_labeling_step}.xyz",
            labeling_step_path / f"labeling_{padded_labeling_step}.xyz",
        )
    )
    atom_count = len(symbols)

    energy = extract_and_convert_energy(
        textfile_to_string_list(
            labeling_step_path / f"2_labeling_{padded_labeling_step}-Force_Eval.fe"
        ),
        np.zeros(1, dtype=np.float64),
        1,
        Ha_to_eV,
        "cp2k",
        program_version,
    )[0]

    box, _, is_periodic = extract_and_convert_box_volume(
        textfile_to_string_list(
            labeling_step_path / f"1_labeling_{padded_labeling_step}.inp"
        ),
        np.zeros((1, 9), dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        1,
        1.0,
        "cp2k",
        program_version,
    )

    force_cp2k = textfile_to_string_list(
        _first_existing(
            labeling_step_path / f"2_naive_labeling_{padded_labeling_step}-Forces.for",
            labeling_step_path / f"2_labeling_{padded_labeling_step}-Forces.for",
        )
    )
    if force_cp2k[-1] == "":
        force_cp2k = force_cp2k[:-1]
    forces = extract_and_convert_forces(
        force_cp2k,
        np.zeros((1, atom_count * 3), dtype=np.float64),
        1,
        au_to_eV_per_A,
        "cp2k",
        program_version,
    ).reshape(atom_count, 3)

    return {
        "symbols": symbols,
        "positions": positions,
        "forces": forces,
        "energy": float(energy),
        "cell": box[0].reshape(3, 3),
        "pbc": bool(is_periodic),
    }


def _frame_text(frame: Dict, config_name: str) -> str:
    """One extended XYZ frame, in the initial datasets' format."""
    lattice = " ".join(repr(float(value)) for value in frame["cell"].flatten())
    pbc = " ".join(["T" if frame["pbc"] else "F"] * 3)
    lines = [
        f"{len(frame['symbols'])}",
        f'Lattice="{lattice}" Properties=species:S:1:pos:R:3:forces_REF:R:3 '
        f'energy_REF={frame["energy"]!r} config={config_name} pbc="{pbc}"',
    ]
    for symbol, position, force in zip(
        frame["symbols"], frame["positions"], frame["forces"]
    ):
        lines.append(
            f"{symbol:<2s} {position[0]:17.8f} {position[1]:17.8f} {position[2]:17.8f}"
            f" {force[0]:17.8f} {force[1]:17.8f} {force[2]:17.8f}"
        )
    return "\n".join(lines) + "\n"


def _extract_system(
    arcann_logger: logging.Logger,
    system_path: Path,
    data_path: Path,
    labeling_steps: range,
    configs_to_skip: Set[int],
    expected_count: int,
    config_prefix: str,
    centroid_module,
    Ha_to_eV: float,
    au_to_eV_per_A: float,
) -> int:
    """Extract ``labeling_steps`` of one system into ``data_path``; returns the frame count."""
    frames = []
    centroid_rows = []
    program_version = None
    for labeling_step in labeling_steps:
        padded_labeling_step = str(labeling_step).zfill(5)
        labeling_step_path = system_path / padded_labeling_step
        if is_config_skipped(labeling_step_path, configs_to_skip):
            continue

        if program_version is None:
            output_cp2k = [
                line
                for line in textfile_to_string_list(
                    labeling_step_path / f"2_labeling_{padded_labeling_step}.out"
                )
                if "CP2K| version string:" in line
            ]
            program_version = float(output_cp2k[0].split()[-1])

        frame = _label_step(
            labeling_step_path,
            padded_labeling_step,
            Ha_to_eV,
            au_to_eV_per_A,
            program_version,
        )

        if centroid_module is not None:
            stride = _cube_stride(centroid_module, labeling_step_path)
            row, warning = (
                centroid_module.centroid_row(
                    labeling_step_path, stride, CENTROID_NEGATIVE
                )
                if stride is not None
                else (None, "no SPIN_DENSITY cube")
            )
            if row is None:
                raise ValueError(
                    f"{labeling_step_path}: {warning}. Fix it, or skip the config "
                    f"(list it in configs_to_skip.txt and rerun 'labeling check')."
                )
            centroid = np.array([row["cx_A"], row["cy_A"], row["cz_A"]], dtype=np.float64)
            # Wrap into [0, L) along the (orthorhombic) cell like the exploration centroid.
            centroid = np.mod(centroid, np.diag(frame["cell"]))
            frame["symbols"].append(ELECTRON_SYMBOL)
            frame["positions"] = np.vstack([frame["positions"], centroid])
            frame["forces"] = np.vstack([frame["forces"], np.zeros(3)])
            centroid_rows.append({**row, "config": f"{config_prefix}_{padded_labeling_step}"})

        frames.append(
            _frame_text(frame, f"{config_prefix}_{padded_labeling_step}")
        )

    if len(frames) != expected_count:
        raise ValueError(
            f"{system_path}: {len(frames)} configs extracted, but 'labeling check' "
            f"counted {expected_count}. Rerun 'labeling check' after changing skips."
        )
    if not frames:
        return 0

    data_path.mkdir(parents=True, exist_ok=True)
    with (data_path / "train.xyz").open("w") as train_file:
        train_file.writelines(frames)

    if centroid_rows:
        with (data_path / "centroid_labels.csv").open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=centroid_module.FIELDS)
            writer.writeheader()
            writer.writerows({key: row[key] for key in centroid_module.FIELDS} for row in centroid_rows)

    arcann_logger.info(
        f"{data_path.name}: {len(frames)} train frames"
        + (" (electron at the spin-density centroid)." if centroid_rows else ".")
    )
    return len(frames)


def extract_mace(
    arcann_logger: logging.Logger,
    current_path: Path,
    training_path: Path,
    padded_curr_iter: str,
    main_json: Dict,
    labeling_json: Dict,
    configs_to_skip: Set[int],
    Ha_to_eV: float,
    au_to_eV_per_A: float,
) -> int:
    """Write every system's labeled configs to ``data/<system>_<iter>/train.xyz``.

    Returns 0 on success, 1 on error (after logging it).
    """
    if labeling_json["labeling_program"] != "cp2k":
        arcann_logger.error("The MACE extraction only supports CP2K labeling. Aborting...")
        return 1

    centroid_module = None
    if main_json.get("hydrated_electron_mode", False):
        try:
            centroid_module = _load_centroid_module(training_path / "user_files")
        except (FileNotFoundError, ImportError) as error:
            arcann_logger.error(f"{error}")
            arcann_logger.error("Aborting...")
            return 1

    for system_auto_index, system_auto in enumerate(labeling_json["systems_auto"]):
        arcann_logger.info(
            f"Processing system: {system_auto} ({system_auto_index + 1}/{len(labeling_json['systems_auto'])})"
        )
        system_json = labeling_json["systems_auto"][system_auto]
        candidates_count = system_json["candidates_count"]
        disturbed_candidates_count = system_json["disturbed_candidates_count"]
        groups = [
            (
                system_auto,
                range(candidates_count),
                candidates_count - system_json["candidates_skipped_count"],
            ),
            (
                f"{system_auto}-disturbed",
                range(candidates_count, candidates_count + disturbed_candidates_count),
                disturbed_candidates_count
                - system_json["disturbed_candidates_skipped_count"],
            ),
        ]
        for name, labeling_steps, expected_count in groups:
            if expected_count == 0:
                arcann_logger.debug(f"No label for {name}, skipping")
                continue
            try:
                _extract_system(
                    arcann_logger,
                    current_path / system_auto,
                    training_path / "data" / f"{name}_{padded_curr_iter}",
                    labeling_steps,
                    configs_to_skip,
                    expected_count,
                    f"{name}_{padded_curr_iter}",
                    centroid_module,
                    Ha_to_eV,
                    au_to_eV_per_A,
                )
            except (OSError, ValueError, IndexError) as error:
                arcann_logger.error(f"{error}")
                arcann_logger.error("Aborting...")
                return 1
        arcann_logger.info(
            f"Processed system: {system_auto} ({system_auto_index + 1}/{len(labeling_json['systems_auto'])})"
        )
    return 0
