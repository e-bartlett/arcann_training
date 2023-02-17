from pathlib import Path
import logging
import sys
import copy
import subprocess

# ### deepmd_iterative imports
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
    load_default_json_file,
    read_key_input_json,
)
from deepmd_iterative.common.file import (
    change_directory,
)

from deepmd_iterative.common.cluster import (
    get_cluster_spec_for_step,
    assert_same_cluster,
)
from deepmd_iterative.common.check import validate_step_folder


def main(
    step_name,
    phase_name,
    deepmd_iterative_path,
    fake_cluster=None,
    input_fn="input.json",
):
    current_path = Path(".").resolve()
    training_path = current_path.parent

    logging.info(f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()}")
    logging.debug(f"Current path :{current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # ### Check if correct folder
    validate_step_folder(step_name)

    # ### Get iteration
    current_iteration_zfill = Path().resolve().parts[-1].split("-")[0]
    current_iteration = int(current_iteration_zfill)

    # ### Get default inputs json
    default_present = False
    default_input_json = load_default_json_file(
        deepmd_iterative_path / "data" / "inputs.json"
    )
    if bool(default_input_json):
        default_present = True

    # ### Get input json (user one)
    if (current_path / input_fn).is_file():
        input_json = load_json_file((current_path / input_fn), True, True)
    else:
        input_json = {}
    new_input_json = copy.deepcopy(input_json)

    # ### Get control path and config_json
    control_apath = training_path / "control"
    config_json = load_json_file((control_apath / "config.json"), True, True)
    training_json = load_json_file(
        (control_apath / f"training_{current_iteration_zfill}.json"), True, True
    )

    # ### Get machine info
    user_spec = read_key_input_json(
        input_json,
        new_input_json,
        "user_spec",
        default_input_json,
        step_name,
        default_present,
    )
    user_spec = None if isinstance(user_spec, bool) else user_spec

    # ### Read cluster info
    (
        cluster,
        cluster_spec,
        cluster_walltime_format,
        cluster_launch_command,
    ) = get_cluster_spec_for_step(
        deepmd_iterative_path,
        training_path,
        "training",
        fake_cluster,
        user_spec,
        True,
    )
    if fake_cluster is not None:
        logging.info(f"Pretending to be on {fake_cluster}")
    else:
        logging.info(f"Cluster is {cluster}")
    del fake_cluster

    # ### Check prep/launch
    assert_same_cluster(cluster, training_json)

    # ### Checks
    if training_json["is_launched"]:
        logging.critical(f"Already launched.")
        continuing = input(
            f"Should it be run again? (Y for Yes, anything else to abort)"
        )
        if continuing == "Y":
            del continuing
        else:
            logging.error(f"Aborting...")
            return 1
    if not training_json["is_locked"]:
        logging.error(f"Lock found. Execute first: training preparation")
        logging.error(f"Aborting...")
        return 1

    # ### Launch the jobs
    check = 0
    for it_nnp in range(1, config_json["nb_nnp"] + 1):
        local_apath = Path(".").resolve() / str(it_nnp)
        if (
            local_apath / f"job_deepmd_train_{training_json['arch_type']}_{cluster}.sh"
        ).is_file():
            change_directory(local_apath)
            try:
                subprocess.call(
                    [
                        training_json["launch_command"],
                        f"./job_deepmd_train_{training_json['arch_type']}_{cluster}.sh",
                    ]
                )
                logging.info(f"DP Train - {it_nnp} launched")
                check = check + 1
            except FileNotFoundError:
                logging.critical(
                    f"DP Train - {it_nnp} NOT launched - {training_json['launch_command']} not found"
                )
            change_directory(local_apath.parent)
        else:
            logging.critical(f"DP Train - {it_nnp} NOT launched - No job file")
        del local_apath
    del it_nnp

    if check == config_json["nb_nnp"]:
        training_json["is_launched"] = True

    write_json_file(
        training_json,
        (control_apath / f"training_{current_iteration_zfill}.json"),
        True,
    )

    logging.info(f"-" * 88)
    if check == config_json["nb_nnp"]:
        logging.info(
            f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a succes !"
        )
    else:
        logging.critical(
            f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is semi-succes !"
        )
        logging.critical(f"Some SLURM jobs did not launch correctly")
        logging.critical(f"Please launch manually before continuing to the next step")
        logging.critical(
            f'Replace the key "is_launched" to True in the training_{current_iteration_zfill}.json.'
        )
    del check

    # ### Cleaning
    del control_apath
    del input_json, default_input_json, default_present, new_input_json
    del config_json
    del current_iteration, current_iteration_zfill
    del training_json
    del cluster
    del training_path, current_path

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "launch",
            Path(sys.argv[1]),
            fake_cluster=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
