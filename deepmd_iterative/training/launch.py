"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/08/30
"""
# Standard library modules
import copy
import logging
import subprocess
import sys
from pathlib import Path

# Local imports
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.filesystem import (
    change_directory,
)
from deepmd_iterative.common.json import (
    backup_and_overwrite_json_file,
    load_default_json_file,
    load_json_file,
    write_json_file,
)
from deepmd_iterative.common.machine import (
    assert_same_machine,
    get_machine_keyword,
    get_machine_spec_for_step,
)


def main(
    current_step: str,
    current_phase: str,
    deepmd_iterative_path: Path,
    fake_machine=None,
    user_config_filename: str = "input.json",
):
    # Get the current path and set the training path as the parent of the current path
    current_path = Path(".").resolve()
    training_path = current_path.parent

    # Log the step and phase of the program
    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}"
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

    # Load the default input JSON
    default_config = load_default_json_file(
        deepmd_iterative_path / "assets" / "default_config.json"
    )[current_step]
    default_config_present = bool(default_config)
    logging.debug(f"default_config: {default_config}")
    logging.debug(f"default_config_present: {default_config_present}")

    # Load the user input JSON
    if (current_path / user_config_filename).is_file():
        user_config = load_json_file((current_path / user_config_filename))
    else:
        user_config = {}
    user_config_present = bool(user_config)
    logging.debug(f"user_config: {user_config}")
    logging.debug(f"user_config_present: {user_config_present}")

    # Make a deepcopy of it to create the current input JSON
    current_config = copy.deepcopy(user_config)

    # Get control path, load the main config JSON and the training config JSON
    control_path = training_path / "control"
    main_config = load_json_file((control_path / "config.json"))
    training_config = load_json_file(
        (control_path / f"training_{padded_curr_iter}.json")
    )

    # Check if we can continue
    if training_config['is_launched']:
        logging.critical(f"Already launched")
        continuing = input(
            f"Should it be run again? (Y for Yes, anything else to abort)"
        )
        if continuing == "Y":
            del continuing
        else:
            logging.error(f"Aborting...")
            return 1
    if not training_config['is_locked']:
        logging.error(f"Lock found. Execute first: training preparation")
        logging.error(f"Aborting...")
        return 1

    # Get the machine keyword (Priority: user > previous > default)
    # And update the current input JSON
    user_machine_keyword = get_machine_keyword(
        user_config, training_config, default_config
    )
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")
    current_config['user_machine_keyword'] = user_machine_keyword
    logging.debug(f"current_config: {current_config}")
    # Set it to None if bool, because: get_machine_spec_for_step needs None
    user_machine_keyword = (
        None if isinstance(user_machine_keyword, bool) else user_machine_keyword
    )
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")

    # From the keyword (or default), get the machine spec (or for the fake one)
    (
        machine,
        machine_spec,
        machine_walltime_format,
        machine_job_scheduler,
        machine_launch_command,
    ) = get_machine_spec_for_step(
        deepmd_iterative_path,
        training_path,
        "training",
        fake_machine,
        user_machine_keyword,
    )
    logging.debug(f"machine: {machine}")
    logging.debug(f"machine_spec: {machine_spec}")
    logging.debug(f"machine_walltime_format: {machine_walltime_format}")
    logging.debug(f"machine_job_scheduler: {machine_job_scheduler}")
    logging.debug(f"machine_launch_command: {machine_launch_command}")

    if fake_machine is not None:
        logging.info(f"Pretending to be on: {fake_machine}")
    else:
        logging.info(f"We are on: {machine}")
    del fake_machine

    # Check prep/launch
    assert_same_machine(machine, training_config)

    # Launch the jobs
    completed_count = 0
    for nnp in range(1, main_config['nnp_count'] + 1):
        local_path = Path(".").resolve() / f"{nnp}"
        if (
            local_path / f"job_deepmd_train_{training_config['arch_type']}_{machine}.sh"
        ).is_file():
            change_directory(local_path)
            try:
                subprocess.run(
                    [
                        training_config['launch_command'],
                        f"./job_deepmd_train_{training_config['arch_type']}_{machine}.sh",
                    ]
                )
                logging.info(f"DP Train - '{nnp}' launched")
                completed_count += 1
            except FileNotFoundError:
                logging.critical(
                    f"DP Train - '{nnp}' NOT launched - '{training_config['launch_command']}' not found"
                )
            change_directory(local_path.parent)
        else:
            logging.critical(f"DP Train - '{nnp}' NOT launched - No job file")
        del local_path
    del nnp

    if completed_count == main_config['nnp_count']:
        training_config['is_launched'] = True

    # Dump the JSON (training config JSON and current input JSON)
    write_json_file(
        training_config, (control_path / f"training_{padded_curr_iter}.json")
    )
    backup_and_overwrite_json_file(
        current_config, (current_path / user_config_filename)
    )

    logging.info(f"-" * 88)
    if completed_count == main_config['nnp_count']:
        logging.info(
            f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
        )
    else:
        logging.critical(
            f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is semi-success!"
        )
        logging.critical(f"Some SLURM jobs did not launch correctly")
        logging.critical(f"Please launch manually before continuing to the next step")
        logging.critical(
            f"Replace the key 'is_launched' to 'True' in the 'training_{padded_curr_iter}'.json"
        )
    del completed_count

    # Cleaning
    del current_path, control_path, training_path
    del (
        default_config,
        default_config_present,
        user_config,
        user_config_present,
        user_config_filename,
    )
    del main_config, current_config, training_config
    del curr_iter, padded_curr_iter
    del machine, machine_spec, machine_walltime_format, machine_launch_command

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "launch",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_config_filename=sys.argv[3],
        )
    else:
        pass
