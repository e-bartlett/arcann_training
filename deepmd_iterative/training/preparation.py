"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/09/18
"""
# Standard library modules
import copy
import logging
import random
import subprocess
import sys
from pathlib import Path

# Non-standard library imports
import numpy as np

# Local imports
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.filesystem import check_directory
from deepmd_iterative.common.json import (
    backup_and_overwrite_json_file,
    get_key_in_dict,
    load_default_json_file,
    load_json_file,
    write_json_file,
)
from deepmd_iterative.common.list import (
    replace_substring_in_string_list,
    string_list_to_textfile,
    textfile_to_string_list,
)
from deepmd_iterative.common.machine import (
    get_machine_keyword,
    get_machine_spec_for_step,
)
from deepmd_iterative.common.slurm import replace_in_slurm_file_general
from deepmd_iterative.training.utils import (
    calculate_decay_rate,
    calculate_decay_steps,
    check_initial_datasets,
    validate_deepmd_config,
    generate_training_json,
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
    logging.debug(f"curr_iter, padded_curr_iter: {curr_iter}, {padded_curr_iter}")

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

    # Get control path and load the main JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))

    # Load the previous training JSON
    if curr_iter > 0:
        prev_iter = curr_iter - 1
        padded_prev_iter = str(prev_iter).zfill(3)
        previous_training_json = load_json_file(
            (control_path / f"training_{padded_prev_iter}.json")
        )
    else:
        previous_training_json = {}

    # Get the machine keyword (Priority: user > previous > default)
    # And update the merged input JSON
    user_machine_keyword = get_machine_keyword(
        user_input_json, previous_training_json, default_input_json, "train"
    )
    logging.debug(f"merged_input_json: {merged_input_json}")
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
        "training",
        fake_machine,
        user_machine_keyword,
    )
    logging.debug(f"machine: {machine}")
    logging.debug(f"machine_walltime_format: {machine_walltime_format}")
    logging.debug(f"machine_job_scheduler: {machine_job_scheduler}")
    logging.debug(f"machine_launch_command: {machine_launch_command}")
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")
    logging.debug(f"machine_spec: {machine_spec}")

    merged_input_json["user_machine_keyword_train"] = user_machine_keyword
    logging.debug(f"merged_input_json: {merged_input_json}")

    if fake_machine is not None:
        logging.info(f"Pretending to be on: '{fake_machine}'.")
    else:
        logging.info(f"Machine identified: '{machine}'.")
    del fake_machine

    # Check if we can continue
    if curr_iter > 0:
        labeling_json = load_json_file(
            (control_path / f"labeling_{padded_curr_iter}.json")
        )
        if not labeling_json["is_extracted"]:
            logging.error(f"Lock found. Run/Check first: labeling extract.")
            logging.error(f"Aborting...")
            return 1
        exploration_json = load_json_file(
            (control_path / f"exploration_{padded_curr_iter}.json")
        )
    else:
        exploration_json = {}
        labeling_json = {}

    # Generate/update both the training JSON and the merged input JSON
    # Priority: user > previous > default
    training_json, merged_input_json = generate_training_json(
        user_input_json,
        previous_training_json,
        default_input_json,
        merged_input_json,
    )
    logging.debug(f"training_json: {training_json}")
    logging.debug(f"merged_input_json: {merged_input_json}")

    # Check if the job file exists
    job_file_name = f"job_deepmd_train_{machine_spec['arch_type']}_{machine}.sh"
    if (current_path.parent / "user_files" / job_file_name).is_file():
        master_job_file = textfile_to_string_list(
            current_path.parent / "user_files" / job_file_name
        )
    else:
        logging.error(
            f"No JOB file provided for '{current_step.capitalize()} / {current_phase.capitalize()}' for this machine."
        )
        logging.error(f"Aborting...")
        return 1

    logging.debug(f"master_job_file: {master_job_file[0:5]}, {master_job_file[-5:-1]}")
    merged_input_json["job_email"] = get_key_in_dict(
        "job_email", user_input_json, previous_training_json, default_input_json
    )
    del job_file_name

    # Check DeePMD version
    validate_deepmd_config(training_json)

    # Check if the default input json file exists
    dp_train_input_path = (
        training_path
        / "user_files"
        / (
            f"dptrain_{training_json['deepmd_model_version']}_{training_json['deepmd_model_type_descriptor']}.json"
        )
    ).resolve()
    dp_train_input = load_json_file(dp_train_input_path)
    main_json["type_map"] = {}
    main_json["type_map"] = dp_train_input["model"]["type_map"]
    del dp_train_input_path
    logging.debug(f"dp_train_input: {dp_train_input}")
    logging.debug(f"main_json: {main_json}")

    # Check the initial sets json file
    initial_datasets_info = check_initial_datasets(training_path)
    logging.debug(f"initial_datasets_info: {initial_datasets_info}")

    # Let us find what is in data
    data_path = training_path / "data"
    check_directory(data_path)

    # This is building the datasets (roughly 200 lines)
    # TODO later
    systems = []
    extra_datasets = []
    validation_datasets = []
    for data_dir in data_path.iterdir():
        if data_dir.is_dir():
            # Escape initial/extra sets, because initial get added first and extra as last, and also escape init_
            # not in initial_json (in case of removal)
            if (
                data_dir.name not in initial_datasets_info.keys()
                and "extra_" != data_dir.name[:6]
                and "init_" != data_dir.name[:5]
            ):
                # Escape test sets
                if "test_" != data_dir.name[:5]:
                    # Escape if set iter is superior as iter, it is only for reprocessing old stuff
                    try:
                        if int(data_dir.name.rsplit("_", 1)[-1]) <= curr_iter:
                            systems.append(data_dir.name.rsplit("_", 1)[0])
                    # TODO Better except clause
                    except:
                        pass
                else:
                    validation_datasets.append(data_dir.name)
            # Get the extra sets !
            elif "extra_" == data_dir.name[:6]:
                extra_datasets.append(data_dir.name)
    del data_dir

    # TODO Implement validation dataset
    del validation_datasets

    # Training sets list construction
    dp_train_input_datasets = []
    training_datasets = []

    # Initial
    initial_count = 0
    if training_json["use_initial_datasets"]:
        for it_datasets_initial_json in initial_datasets_info.keys():
            if (data_path / it_datasets_initial_json).is_dir():
                dp_train_input_datasets.append(
                    f"{(Path(data_path.parts[-1]) / it_datasets_initial_json / '_')}"[
                        :-1
                    ]
                )
                training_datasets.append(it_datasets_initial_json)
                initial_count += initial_datasets_info[it_datasets_initial_json]

        del it_datasets_initial_json
    del initial_datasets_info

    # This trick remove duplicates from list via set
    systems = list(set(systems))
    systems = [i for i in systems if i not in main_json["systems_auto"]]
    systems = [
        i
        for i in systems
        if i not in [zzz + "-disturbed" for zzz in main_json["systems_auto"]]
    ]
    systems = sorted(systems)
    main_json["systems_adhoc"] = systems
    del systems

    # TODO As function
    # Automatic Systems (aka systems_auto in the initialization first) && all the others are not automated !
    # Total and what is added just for this iteration
    added_auto_count = 0
    added_adhoc_count = 0
    added_auto_iter_count = 0
    added_adhoc_iter_count = 0

    if curr_iter > 0:
        for iteration in np.arange(1, curr_iter + 1):
            padded_iteration = str(iteration).zfill(3)
            try:
                for system_auto in ["systems_auto"]:
                    if (data_path / f"{system_auto}_{padded_iteration}").is_dir():
                        dp_train_input_datasets.append(
                            f"{(Path(data_path.parts[-1]) / (system_auto+'_'+padded_iteration) / '_')}"[
                                :-1
                            ]
                        )
                        training_datasets.append(f"{system_auto}_{padded_iteration}")
                        added_auto_count += np.load(
                            data_path
                            / f"{system_auto}_{padded_iteration}"
                            / "set.000"
                            / "box.npy"
                        ).shape[0]
                        if iteration == curr_iter:
                            added_auto_iter_count += np.load(
                                data_path
                                / f"{system_auto}_{padded_iteration}"
                                / "set.000"
                                / "box.npy"
                            ).shape[0]
                del system_auto
            except (KeyError, NameError):
                pass
            try:
                for system_auto_disturbed in [
                    zzz + "-disturbed" for zzz in main_json["systems_auto"]
                ]:
                    if (
                        data_path / f"{system_auto_disturbed}_{padded_iteration}"
                    ).is_dir():
                        dp_train_input_datasets.append(
                            f"{(Path(data_path.parts[-1]) / (system_auto_disturbed+'_'+padded_iteration) / '_')}"[
                                :-1
                            ]
                        )
                        training_datasets.append(
                            f"{system_auto_disturbed}_{padded_iteration}"
                        )
                        added_auto_count += np.load(
                            data_path
                            / f"{system_auto_disturbed}_{padded_iteration}"
                            / "set.000"
                            / "box.npy"
                        ).shape[0]
                        if iteration == curr_iter:
                            added_auto_iter_count += np.load(
                                data_path
                                / f"{system_auto_disturbed}_{padded_iteration}"
                                / "set.000"
                                / "box.npy"
                            ).shape[0]
                del system_auto_disturbed
            except (KeyError, NameError):
                pass
            try:
                for system_adhoc in main_json["systems_adhoc"]:
                    if (data_path / f"{system_adhoc}_{padded_iteration}").is_dir():
                        dp_train_input_datasets.append(
                            f"{(Path(data_path.parts[-1]) / (system_adhoc+'_'+padded_iteration) / '_')}"[
                                :-1
                            ]
                        )
                        training_datasets.append(f"{system_adhoc}_{padded_iteration}")
                        added_auto_count = (
                            added_auto_count
                            + np.load(
                                data_path
                                / f"{system_adhoc}_{padded_iteration}"
                                / "set.000"
                                / "box.npy"
                            ).shape[0]
                        )
                        if iteration == curr_iter:
                            added_auto_iter_count += np.load(
                                data_path
                                / f"{system_adhoc}_{padded_iteration}"
                                / "set.000"
                                / "box.npy"
                            ).shape[0]
                del system_adhoc
            except (KeyError, NameError):
                pass
        del iteration, padded_iteration
    # TODO End of As function

    # Finally the extra sets !
    extra_count = 0
    if training_json["use_extra_datasets"]:
        main_json["extra_datasets"] = extra_datasets
        del extra_datasets
        for extra_dataset in main_json["extra_datasets"]:
            dp_train_input_datasets.append(
                f"{(Path(data_path.parts[-1]) / extra_dataset / '_')}"[:-1]
            )
            training_datasets.append(extra_dataset)
            extra_count += np.load(
                data_path / extra_dataset / "set.000" / "box.npy"
            ).shape[0]
        del extra_dataset
    else:
        del extra_datasets

    # Total
    trained_count = initial_count + added_auto_count + added_adhoc_count + extra_count
    logging.debug(
        f"trained_count: {trained_count} = {initial_count} + {added_auto_count} + {added_adhoc_count} + {extra_count}"
    )
    logging.debug(f"dp_train_input_datasets: {dp_train_input_datasets}")

    # Update the inputs with the sets
    dp_train_input["training"]["training_data"]["systems"] = dp_train_input_datasets

    # Update the training JSON
    training_json = {
        **training_json,
        "training_datasets": training_datasets,
        "trained_count": trained_count,
        "initial_count": initial_count,
        "added_auto_count": added_auto_count,
        "added_adhoc_count": added_adhoc_count,
        "added_auto_iter_count": added_auto_iter_count,
        "added_adhoc_iter_count": added_adhoc_iter_count,
        "extra_count": extra_count,
    }
    logging.debug(f"training_json: {training_json}")

    del training_datasets
    del trained_count, initial_count, extra_count
    del (
        added_auto_count,
        added_adhoc_count,
        added_auto_iter_count,
        added_adhoc_iter_count,
    )

    # Here calculate the parameters
    # decay_steps it auto-recalculated as funcion of trained_count
    logging.debug(f"training_json - decay_steps: {training_json['decay_steps']}")
    logging.debug(
        f"merged_input_json - decay_steps: {merged_input_json['decay_steps']}"
    )
    if not training_json["decay_steps_fixed"]:
        decay_steps = calculate_decay_steps(
            training_json["trained_count"], training_json["decay_steps"]
        )
        logging.debug(f"Recalculating decay_steps")
        # Update the training JSON and the merged input JSON
        training_json["decay_steps"] = decay_steps
        merged_input_json["decay_steps"] = decay_steps
    else:
        decay_steps = training_json["decay_steps"]
    logging.debug(f"decay_steps: {decay_steps}")
    logging.debug(f"training_json - decay_steps: {training_json['decay_steps']}")
    logging.debug(
        f"merged_input_json - decay_steps: {merged_input_json['decay_steps']}"
    )

    # numb_steps and decay_rate
    logging.debug(
        f"training_json - numb_steps / decay_rate: {training_json['numb_steps']} / {training_json['decay_rate']}"
    )
    logging.debug(
        f"merged_input_json - numb_steps / decay_rate: {merged_input_json['numb_steps']} / {merged_input_json['decay_rate']}"
    )
    numb_steps = training_json["numb_steps"]
    decay_rate_new = calculate_decay_rate(
        numb_steps,
        training_json["start_lr"],
        training_json["stop_lr"],
        training_json["decay_steps"],
    )
    while decay_rate_new < training_json["decay_rate"]:
        numb_steps = numb_steps + 10000
        decay_rate_new = calculate_decay_rate(
            numb_steps,
            training_json["start_lr"],
            training_json["stop_lr"],
            training_json["decay_steps"],
        )
    # Update the training JSON and the merged input JSON
    training_json["numb_steps"] = int(numb_steps)
    training_json["decay_rate"] = decay_rate_new
    merged_input_json["numb_steps"] = int(numb_steps)
    merged_input_json["decay_rate"] = decay_rate_new
    logging.debug(f"numb_steps: {numb_steps}")
    logging.debug(f"decay_rate: {decay_rate_new}")
    logging.debug(
        f"training_json - numb_steps / decay_rate: {training_json['numb_steps']} / {training_json['decay_rate']}"
    )
    logging.debug(
        f"merged_input_json - numb_steps / decay_rate: {merged_input_json['numb_steps']} / {merged_input_json['decay_rate']}"
    )

    del decay_steps, numb_steps, decay_rate_new

    dp_train_input["training"]["numb_steps"] = training_json["numb_steps"]
    dp_train_input["learning_rate"]["decay_steps"] = training_json["decay_steps"]
    dp_train_input["learning_rate"]["stop_lr"] = training_json["stop_lr"]

    # Set booleans in the training JSON
    training_json = {
        **training_json,
        "is_locked": True,
        "is_launched": False,
        "is_checked": False,
        "is_frozen": False,
        "is_compressed": False,
        "is_incremented": False,
    }

    # Rsync data to local data
    localdata_path = current_path / "data"
    localdata_path.mkdir(exist_ok=True)
    for dp_train_input_dataset in dp_train_input_datasets:
        subprocess.run(
            [
                "rsync",
                "-a",
                f"{training_path / (dp_train_input_dataset.rsplit('/', 1)[0])}",
                f"{localdata_path}",
            ]
        )
    del dp_train_input_dataset, localdata_path, dp_train_input_datasets

    # Change some inside output
    dp_train_input["training"]["disp_file"] = "lcurve.out"
    dp_train_input["training"]["save_ckpt"] = "model.ckpt"

    logging.debug(f"training_json: {training_json}")
    logging.debug(f"user_input_json: {user_input_json}")
    logging.debug(f"merged_input_json: {merged_input_json}")
    logging.debug(f"default_input_json: {default_input_json}")
    logging.debug(f"previous_training_json: {previous_training_json}")

    # Create the inputs/jobfiles for each NNP with random SEED

    # Walltime
    if curr_iter == 0:
        if (
            "init_job_walltime_train_h" in user_input_json
            and user_input_json["init_job_walltime_train_h"] > 0
        ):
            walltime_approx_s = int(user_input_json["init_job_walltime_train_h"] * 3600)
            logging.debug(
                f"init_job_walltime_train_h: {user_input_json['init_job_walltime_train_h']}"
            )
        elif (
            "job_walltime_train_h" in user_input_json
            and user_input_json["job_walltime_train_h"] > 0
        ):
            walltime_approx_s = int(user_input_json["job_walltime_train_h"] * 3600)
            logging.debug(
                f"job_walltime_train_h: {user_input_json['job_walltime_train_h']}"
            )
        elif (
            "mean_s_per_step" in user_input_json
            and user_input_json["mean_s_per_step"] > 0
        ):
            walltime_approx_s = int(
                np.ceil(
                    (training_json["numb_steps"] * user_input_json["mean_s_per_step"])
                )
            )
            logging.debug(f"mean_s_per_step: {user_input_json['mean_s_per_step']}")
        else:
            walltime_approx_s = int(
                max(
                    np.ceil(
                        training_json["numb_steps"]
                        * default_input_json["mean_s_per_step"]
                    ),
                    default_input_json["init_job_walltime_train_h"] * 3600,
                    default_input_json["job_walltime_train_h"] * 3600,
                )
            )
            merged_input_json["init_job_walltime_train_h"] = -1
            merged_input_json["job_walltime_train_h"] = -1
            merged_input_json["mean_s_per_step"] = -1
    else:
        if (
            "job_walltime_train_h" in user_input_json
            and user_input_json["job_walltime_train_h"] > 0
        ):
            walltime_approx_s = int(user_input_json["job_walltime_train_h"] * 3600)
            logging.debug(
                f"job_walltime_train_h: {user_input_json['job_walltime_train_h']}"
            )
        elif (
            "mean_s_per_step" in user_input_json
            and user_input_json["mean_s_per_step"] > 0
        ):
            walltime_approx_s = int(
                np.ceil(
                    (training_json["numb_steps"] * user_input_json["mean_s_per_step"])
                )
            )
            logging.debug(f"mean_s_per_step: {user_input_json['mean_s_per_step']}")
        else:
            walltime_approx_s = int(
                np.ceil(
                    (
                        training_json["numb_steps"]
                        * (previous_training_json["mean_s_per_step"] * 1.50)
                    )
                )
            )
            logging.debug(
                f"mean_s_per_step: {previous_training_json['mean_s_per_step']}"
            )
            merged_input_json["job_walltime_train_h"] = -1
            merged_input_json["mean_s_per_step"] = -1

    logging.debug(f"walltime_approx_s: {walltime_approx_s}")

    for nnp in range(1, main_json["nnp_count"] + 1):
        local_path = current_path / f"{nnp}"
        local_path.mkdir(exist_ok=True)
        check_directory(local_path)

        random.seed()
        random_0_1000 = random.randrange(0, 1000)

        if training_json["deepmd_model_type_descriptor"] == "se_ar":
            dp_train_input["model"]["descriptor"]["a"]["seed"] = int(
                f"{nnp}{random_0_1000}{padded_curr_iter}"
            )
            dp_train_input["model"]["descriptor"]["r"]["seed"] = int(
                f"{nnp}{random_0_1000}{padded_curr_iter}"
            )
        else:
            dp_train_input["model"]["descriptor"]["seed"] = int(
                f"{nnp}{random_0_1000}{padded_curr_iter}"
            )
        dp_train_input["model"]["fitting_net"]["seed"] = int(
            f"{nnp}{random_0_1000}{padded_curr_iter}"
        )
        dp_train_input["training"]["seed"] = int(
            f"{nnp}{random_0_1000}{padded_curr_iter}"
        )

        dp_train_input_file = (Path(f"{nnp}") / "training.json").resolve()

        write_json_file(dp_train_input, dp_train_input_file, False)

        job_file = replace_in_slurm_file_general(
            master_job_file,
            machine_spec,
            walltime_approx_s,
            machine_walltime_format,
            training_json["job_email"],
        )

        job_file = replace_substring_in_string_list(
            job_file, "_R_DEEPMD_VERSION_", f"{training_json['deepmd_model_version']}"
        )
        # TODO This feature is not used. Write a way to if the training didn't finish, restart it and relaunch. (probably in check.py, and ask user for confirmation)
        job_file = replace_substring_in_string_list(
            job_file, "_R_CHECKPOINT_", f"model.ckpt"
        )

        string_list_to_textfile(
            local_path / f"job_deepmd_train_{machine_spec['arch_type']}_{machine}.sh",
            job_file,
        )
        del job_file, local_path, dp_train_input_file, random_0_1000

    del nnp, walltime_approx_s, dp_train_input

    # Dump the JSON files (main, training and merged input)
    logging.info(f"-" * 88)
    write_json_file(main_json, (control_path / "config.json"))
    write_json_file(training_json, (control_path / f"training_{padded_curr_iter}.json"))
    backup_and_overwrite_json_file(
        merged_input_json, (current_path / user_input_json_filename)
    )

    # End
    logging.info(f"-" * 88)
    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
    )

    # Cleaning
    del current_path, control_path, training_path, data_path
    del (
        default_input_json,
        default_input_json_present,
        user_input_json,
        user_input_json_present,
        user_input_json_filename,
    )
    del (
        main_json,
        merged_input_json,
        training_json,
        previous_training_json,
        labeling_json,
    )
    del user_machine_keyword
    del curr_iter, padded_curr_iter
    del (
        machine,
        machine_spec,
        machine_walltime_format,
        machine_launch_command,
        machine_job_scheduler,
    )
    del master_job_file

    logging.debug(f"LOCAL")
    logging.debug(f"{locals()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "preparation",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass
