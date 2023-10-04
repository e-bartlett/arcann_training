"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/10/04
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

    # Load the default input JSON
    default_input_json = load_default_json_file(
        deepmd_iterative_path / "assets" / "default_config.json"
    )[current_step]
    default_input_json_present = bool(default_input_json)
    logging.debug(f"default_input_json: {default_input_json}")
    logging.debug(f"default_input_json_present: {default_input_json_present}")

    # Load the user input JSON
    if (current_path / user_input_json_filename).is_file():
        user_input_json = load_json_file((current_path / user_input_json_filename))
    else:
        user_input_json = {}
    user_input_json_present = bool(user_input_json)
    logging.debug(f"user_input_json: {user_input_json}")
    logging.debug(f"user_input_json_present: {user_input_json_present}")

    # Make a deepcopy of it to create the merged input JSON
    merged_input_json = copy.deepcopy(user_input_json)

    # Get control path, load the main JSON and the exploration JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))
    exploration_json = load_json_file(
        (control_path / f"exploration_{padded_curr_iter}.json")
    )

    # Get the machine keyword (Priority: user > previous > default)
    # And update the merged input JSON
    user_machine_keyword = get_machine_keyword(
        user_input_json, exploration_json, default_input_json, "exp"
    )
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")
    # Set it to None if bool, because: get_machine_spec_for_step needs None
    user_machine_keyword = (
        None if isinstance(user_machine_keyword, bool) else user_machine_keyword
    )
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")

    # From the keyword (or default), get the machine spec (or for the fake one)
    (
        machine,
        machine_walltime_format,
        machine_job_scheduler,
        machine_launch_command,
        user_machine_keyword,
        machine_spec,
    ) = get_machine_spec_for_step(
        deepmd_iterative_path,
        training_path,
        "exploration",
        fake_machine,
        user_machine_keyword,
    )
    arch_type = machine_spec['arch_type']
    logging.debug(f"machine: {machine}")
    logging.debug(f"machine_walltime_format: {machine_walltime_format}")
    logging.debug(f"machine_job_scheduler: {machine_job_scheduler}")
    logging.debug(f"machine_launch_command: {machine_launch_command}")
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")
    logging.debug(f"machine_spec: {machine_spec}")

    merged_input_json["user_machine_keyword_exp"] = user_machine_keyword
    logging.debug(f"merged_input_json: {merged_input_json}")

    if fake_machine is not None:
        logging.info(f"Pretending to be on: '{fake_machine}'.")
    else:
        logging.info(f"Machine identified: '{machine}'.")
    del fake_machine

    # Check prep/launch
    assert_same_machine(user_machine_keyword, exploration_json, "exp")

    # Check if we can continue
    if exploration_json["is_launched"]:
        logging.critical(f"Already launched...")
        continuing = input(
            f"Do you want to continue?\n['Y' for yes, anything else to abort]\n"
        )
        if continuing == "Y":
            del continuing
        else:
            logging.error(f"Aborting...")
            return 1
    if not exploration_json["is_locked"]:
        logging.error(f"Lock found. Execute first: exploration preparation.")
        logging.error(f"Aborting...")
        return 1

    # Launch the jobs
    exploration_types = []
    for system_auto in exploration_json["systems_auto"]:
        exploration_types.append(exploration_json['systems_auto'][system_auto]['exploration_type'])
    exploration_types = list(set(exploration_types))

    completed_count = 0
    for exploration_type in exploration_types:
        job_name = f"job-array_{exploration_type}-deepmd_{arch_type}_{machine}.sh"
        if job_name.is_file():
            subprocess.run([machine_launch_command, f"./job-array_{exploration_type}-deepmd_{arch_type}_{machine}.sh"])
            logging.info(f"Exploration - Array LAMMPS launched.")
            completed_count += 1

    # for system_auto_index, system_auto in enumerate(exploration_json["systems_auto"]):
    #     for nnp_index in range(1, main_json["nnp_count"] + 1):
    #         for traj_index in range(
    #             1, exploration_json["systems_auto"][system_auto]["traj_count"] + 1
    #         ):
    #             local_path = (
    #                 Path(".").resolve()
    #                 / str(system_auto)
    #                 / str(nnp_index)
    #                 / (str(traj_index).zfill(5))
    #             )

    #             if (
    #                 local_path
    #                 / f"job_deepmd_{exploration_json['systems_auto'][system_auto]['exploration_type']}_{machine_spec['arch_type']}_{machine}.sh"
    #             ).is_file():
    #                 change_directory(local_path)
    #                 try:
    #                     subprocess.run(
    #                         [
    #                             machine_launch_command,
    #                             f"./job_deepmd_{exploration_json['systems_auto'][system_auto]['exploration_type']}_{machine_spec['arch_type']}_{machine}.sh",
    #                         ]
    #                     )
    #                     logging.info(
    #                         f"Exploration - '{system_auto}' '{nnp_index}' '{traj_index}' launched."
    #                     )
    #                     completed_count += 1
    #                 except FileNotFoundError:
    #                     logging.critical(
    #                         f"Exploration - '{system_auto}' '{nnp_index}' '{traj_index}' NOT launched - '{machine_launch_command}' not found."
    #                     )
    #                 change_directory(local_path.parent.parent.parent)
    #             else:
    #                 logging.critical(
    #                     f"Exploration - '{system_auto}' '{nnp_index}' '{traj_index}' NOT launched - No job file."
    #                 )
    #             del local_path
    #         del traj_index
    #     del nnp_index
    # del system_auto_index, system_auto

    logging.info(f"-" * 88)
    # Update the booleans in the exploration JSON
    # if completed_count == (
    #     exploration_json["nnp_count"]
    #     * sum(
    #         [
    #             exploration_json["systems_auto"][_]["traj_count"]
    #             for _ in exploration_json["systems_auto"]
    #         ]
    #     )
    # ):
    if completed_count == len(exploration_types):
        exploration_json["is_launched"] = True

    # Dump the JSON files (exploration JSON and merged input JSON)
    write_json_file(
        exploration_json, (control_path / f"exploration_{padded_curr_iter}.json")
    )
    backup_and_overwrite_json_file(
        merged_input_json, (current_path / user_input_json_filename)
    )

    # End
    logging.info(f"-" * 88)
    # if completed_count == (
    #     exploration_json["nnp_count"]
    #     * sum(
    #         [
    #             exploration_json["systems_auto"][_]["traj_count"]
    #             for _ in exploration_json["systems_auto"]
    #         ]
    #     )
    # ):
    if completed_count == len(exploration_types):
        logging.info(
            f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
        )
    else:
        logging.critical(
            f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is semi-success!"
        )
        logging.critical(f"Some jobs did not launch correctly.")
        logging.critical(f"Please launch manually before continuing to the next step.")
        logging.critical(
            f"Replace the key 'is_launched' to 'True' in the 'exploration_{padded_curr_iter}.json'."
        )
    del completed_count

    # Cleaning
    del current_path, control_path, training_path
    del (
        default_input_json,
        default_input_json_present,
        user_input_json,
        user_input_json_present,
        user_input_json_filename,
    )
    del main_json, merged_input_json
    del curr_iter, padded_curr_iter
    del machine, machine_spec, machine_walltime_format, machine_launch_command

    logging.debug(f"LOCAL")
    logging.debug(f"{locals()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "exploration",
            "launch",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass
