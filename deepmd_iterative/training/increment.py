"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/09/19
"""
# Standard library modules
import logging
import subprocess
import sys
from pathlib import Path

# Local imports
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.filesystem import (
    check_directory,
    check_file_existence,
    remove_file,
    remove_files_matching_glob,
    remove_tree,
)
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
)


def main(
    current_step: str,
    current_phase: str,
    deepmd_iterative_path: Path,
    fake_machine=None,
    user_input_json_filename: str = "input.json",
):
    # Get the current path and set the training path as the parent of the current path
    current_path = Path(".").resolve()
    training_path = current_path.parent

    # Log the step and phase of the program
    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}."
    )
    logging.debug(f"Current path :{current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # Check if the current folder is correct for the current step
    validate_step_folder(current_step)

    # Get the current iteration number
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)

    # Get control path, load the main JSON and the training JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))
    training_json = load_json_file((control_path / f"training_{padded_curr_iter}.json"))

    # Check if we can continue
    if not training_json["is_frozen"]:
        logging.error(f"Lock found. Execute first: training check_freeze.")
        logging.error(f"Aborting...")
        return 1

    # Check if pb files are present and delete temp files
    for nnp in range(1, main_json["nnp_count"] + 1):
        local_path = Path(".").resolve() / f"{nnp}"
        check_file_existence(local_path / f"graph_{nnp}_{padded_curr_iter}.pb")
        if training_json["is_compressed"]:
            check_file_existence(
                local_path / f"graph_{nnp}_{padded_curr_iter}_compressed.pb"
            )

    # Prepare the test folder
    (training_path / f"{padded_curr_iter}-test").mkdir(exist_ok=True)
    check_directory((training_path / f"{padded_curr_iter}-test"))

    subprocess.run(
        [
            "rsync",
            "-a",
            f"{training_path / 'data'}",
            str(training_path / f"{padded_curr_iter}-test"),
        ]
    )

    # Copy the pb files to the NNP meta folder
    (training_path / "NNP").mkdir(exist_ok=True)
    check_directory(training_path / "NNP")

    local_path = Path(".").resolve()

    for nnp in range(1, main_json["nnp_count"] + 1):
        if training_json["is_compressed"]:
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    str(
                        local_path
                        / f"{nnp}"
                        / f"graph_{nnp}_{padded_curr_iter}_compressed.pb"
                    ),
                    str((training_path / "NNP")),
                ]
            )
        subprocess.run(
            [
                "rsync",
                "-a",
                str(local_path / f"{nnp}" / f"graph_{nnp}_{padded_curr_iter}.pb"),
                str((training_path / "NNP")),
            ]
        )
    del nnp

    # Next iteration
    next_iter = curr_iter + 1
    main_json["curr_iter"] = next_iter
    padded_next_iter = str(next_iter).zfill(3)

    for step in ["exploration", "adhoc", "labeling", "training"]:
        (training_path / f"{padded_next_iter}-{step}").mkdir(exist_ok=True)
        check_directory(training_path / f"{padded_next_iter}-{step}")
    del step

    logging.info(f"-" * 88)
    # Update the boolean in the training JSON
    training_json["is_incremented"] = True

    # Dump the JSON files (main, training)
    write_json_file(training_json, (control_path / f"training_{padded_curr_iter}.json"))
    write_json_file(main_json, (control_path / "config.json"))

    # End
    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
    )

    # Cleaning
    del current_path, control_path, training_path
    del user_input_json_filename
    del main_json, training_json
    del curr_iter, padded_curr_iter, next_iter, padded_next_iter

    logging.debug(f"LOCAL")
    logging.debug(f"{locals()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "update_iter",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass
