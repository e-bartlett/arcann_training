from pathlib import Path
import logging
import sys
import copy
import subprocess
import random

# ### Non-standard imports
import numpy as np

# ### deepmd_iterative imports
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
    backup_and_overwrite_json_file,
    load_default_json_file,
    read_key_input_json,
)
from deepmd_iterative.common.list import replace_substring_in_list_of_strings
from deepmd_iterative.common.xml import parse_xml_file, convert_xml_to_list_of_strings
from deepmd_iterative.common.cluster import get_cluster_spec_for_step
from deepmd_iterative.common.file import (
    check_file_existence,
    file_to_list_of_strings,
    check_directory,
    write_list_of_strings_to_file,
)

from deepmd_iterative.common.slurm import replace_in_slurm_file
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
    control_path = training_path / "control"
    config_json = load_json_file((control_path / "config.json"), True, True)
    previous_iteration_zfill = str(current_iteration - 1).zfill(3)
    prevtraining_json = load_json_file(
        (control_path / ("training_" + previous_iteration_zfill + ".json")), True, True
    )
    if int(previous_iteration_zfill) > 0:
        prevexploration_json = load_json_file(
            (control_path / ("exploration_" + previous_iteration_zfill + ".json")),
            True,
            True,
        )

    # ### Get extra needed paths
    jobs_path = deepmd_iterative_path / "data" / "jobs" / "exploration"

    # ### Get user cluster keyword
    user_cluster_keyword = read_key_input_json(
        input_json,
        new_input_json,
        "user_cluster_keyword",
        default_input_json,
        step_name,
        default_present,
    )
    user_cluster_keyword = (
        None if isinstance(user_cluster_keyword, bool) else user_cluster_keyword
    )

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
        user_cluster_keyword,
    )
    if fake_cluster is not None:
        logging.info(f"Pretending to be on {fake_cluster}")
    else:
        logging.info(f"Cluster is {cluster}")
    del fake_cluster

    # # ### Checks
    # if current_iteration > 0:
    #     labeling_json = load_json_file(
    #         (control_path / f"labeling_{current_iteration_zfill}.json"), True, True
    #     )
    #     if not labeling_json["is_extracted"]:
    #         logging.error("Lock found. Execute first: training update_iter")
    #         logging.error("Aborting...")
    #         return 1

    # ### Get/Create exploration parameters
    exploration_json = load_json_file(
        (control_path / f"exploration_{current_iteration_zfill}.json"), False, True
    )

    exploration_json["deepmd_model_version"] = prevtraining_json["deepmd_model_version"]
    exploration_json["nb_nnp"] = config_json["nb_nnp"]
    exploration_json["exploration_type"] = config_json["exploration_type"]
    exploration_json["nb_traj"] = read_key_input_json(
        input_json,
        new_input_json,
        "nb_traj",
        default_input_json,
        step_name,
        default_present,
    )

    exploration_json["cluster"] = cluster
    exploration_json["project_name"] = cluster_spec["project_name"]
    exploration_json["allocation_name"] = cluster_spec["allocation_name"]
    exploration_json["arch_name"] = cluster_spec["arch_name"]
    exploration_json["arch_type"] = cluster_spec["arch_type"]
    exploration_json["launch_command"] = cluster_launch_command

    check_file_existence(
        jobs_path
        / (
            "job_deepmd_"
            + exploration_json["exploration_type"]
            + "_"
            + exploration_json["arch_type"]
            + "_"
            + cluster
            + ".sh"
        ),
        True,
        True,
        "No SLURM file present for the exploration step on this cluster.",
    )
    slurm_file_master = file_to_list_of_strings(
        jobs_path
        / (
            "job_deepmd_"
            + exploration_json["exploration_type"]
            + "_"
            + exploration_json["arch_type"]
            + "_"
            + cluster
            + ".sh"
        )
    )
    del jobs_path

    ### Preparation of the exploration
    exploration_json["subsys_nr"] = {}

    for it0_subsys_nr, it_subsys_nr in enumerate(config_json["subsys_nr"]):

        random.seed()
        exploration_json["subsys_nr"][it_subsys_nr] = {}

        with_plumed = 0
        with_plumed_smd = 0
        with_plumed_smd_nb_steps = 0

        # ### Get the input file (.in or .xml) first and check if plumed
        if exploration_json["exploration_type"] == "lammps":
            exploration_type = 0
            subsys_lammps_in = file_to_list_of_strings(
                training_path / "files" / (it_subsys_nr + ".in")
            )
            if any("plumed" in zzz for zzz in subsys_lammps_in):
                with_plumed = 1
        elif exploration_json["exploration_type"] == "i-PI":
            exploration_type = 1
            subsys_ipi_xml = parse_xml_file(
                training_path / "files" / (it_subsys_nr + ".xml")
            )
            subsys_ipi_xml_aslist = convert_xml_to_list_of_strings(subsys_ipi_xml)
            subsys_ipi_json = {
                "verbose": False,
                "use_unix": False,
                "port": "_R_NB_PORT_",
                "host": "_R_ADDRESS_",
                "graph_file": "_R_GRAPH_",
                "coord_file": "_R_XYZ_",
                "atom_type": {},
            }
            if any("plumed" in zzz for zzz in subsys_ipi_xml_aslist):
                with_plumed = 1

        # ### Get plumed files
        if with_plumed == 1:
            plumed_files_list = [
                zzz
                for zzz in (training_path / "files").glob(
                    "plumed*_" + it_subsys_nr + ".dat"
                )
            ]
            if len(plumed_files_list) == 0:
                logging.critical("Plumed in input but no plumed files")
                logging.critical("Aborting...")
                sys.exit(1)
            plumed_input = {}
            for it_plumed_files_list in plumed_files_list:
                plumed_input[it_plumed_files_list.name] = file_to_list_of_strings(
                    it_plumed_files_list
                )
                if any(
                    "MOVINGRESTRAINT" in zzz
                    for zzz in plumed_input[it_plumed_files_list.name]
                ):
                    SMD_steps = [
                        zzz
                        for zzz in plumed_input[it_plumed_files_list.name]
                        if "STEP" in zzz
                    ]
                    with_plumed_smd = 1
                    with_plumed_smd_nb_steps = SMD_steps[-1].split("=")[1].split(" ")[0]
                    del SMD_steps
            del it_plumed_files_list, plumed_files_list

        # ### Timestep
        subsys_timestep = read_key_input_json(
            input_json,
            new_input_json,
            "timestep_ps",
            default_input_json,
            step_name,
            default_present,
            subsys_index=it0_subsys_nr,
            subsys_number=len(config_json["subsys_nr"]),
            exploration_dep=exploration_type,
        )

        # ### Temperature
        subsys_temp = read_key_input_json(
            input_json,
            new_input_json,
            "temperature_K",
            default_input_json,
            step_name,
            default_present,
            subsys_index=it0_subsys_nr,
            subsys_number=len(config_json["subsys_nr"]),
            exploration_dep=exploration_type,
        )

        if current_iteration == 1:
            # ### Initial Exploration Time
            subsys_init_exp_time_ps = read_key_input_json(
                input_json,
                new_input_json,
                "initial_exploration_time_ps",
                default_input_json,
                step_name,
                default_present,
                subsys_index=it0_subsys_nr,
                subsys_number=len(config_json["subsys_nr"]),
                exploration_dep=exploration_type,
            )

            # ### Initial Wall Time
            subsys_init_walltime_h = read_key_input_json(
                input_json,
                new_input_json,
                "init_job_walltime_h",
                default_input_json,
                step_name,
                default_present,
                subsys_index=it0_subsys_nr,
                subsys_number=len(config_json["subsys_nr"]),
                exploration_dep=exploration_type,
            )

        # ### LAMMPS Input
        if exploration_type == 0:
            subsys_lammps_in = replace_substring_in_list_of_strings(
                subsys_lammps_in, "_R_TEMPERATURE_", str(subsys_temp)
            )
            subsys_lammps_in = replace_substring_in_list_of_strings(
                subsys_lammps_in, "_R_TIMESTEP_", str(subsys_timestep)
            )
            # ### First exploration
            if current_iteration == 1:
                subsys_lammps_data_fn = it_subsys_nr + ".lmp"
                subsys_lammps_data = file_to_list_of_strings(
                    training_path / "files" / subsys_lammps_data_fn
                )
                subsys_lammps_in = replace_substring_in_list_of_strings(
                    subsys_lammps_in, "_R_DATA_FILE_", subsys_lammps_data_fn
                )
                # ### Default time and nb steps
                if with_plumed_smd == 1:
                    subsys_nb_steps = int(with_plumed_smd_nb_steps)
                else:
                    subsys_nb_steps = subsys_init_exp_time_ps / subsys_timestep
                subsys_lammps_in = replace_substring_in_list_of_strings(
                    subsys_lammps_in, "_R_NUMBER_OF_STEPS_", str(subsys_nb_steps)
                )
                subsys_walltime_approx_s = subsys_init_walltime_h * 3600
                # ### Get the cell and nb of atoms (just for config.json)
                subsys_cell, subsys_nb_atm = get_cell_nbatoms_from_lmp(
                    subsys_lammps_data
                )

        # ### i-PI // UNTESTED
        elif exploration_type == 1:
            # subsys_temp = float(get_temp_from_xml_tree(subsys_ipi_xml))
            if subsys_temp == -1:
                logging.critical("No temperature found in the xml")
                logging.critical("Aborting...")
                sys.exit(1)
            subsys_ipi_xml_aslist = replace_substring_in_list_of_strings(
                subsys_ipi_xml_aslist, "_R_TIMESTEP_", str(subsys_timestep)
            )
            subsys_ipi_xml_aslist = replace_substring_in_list_of_strings(
                subsys_ipi_xml_aslist, "_R_NB_BEADS_", str(8)
            )
            # ### First exploration
            if current_iteration == 1:
                subsys_lammps_data_fn = it_subsys_nr + ".lmp"
                subsys_lammps_data = file_to_list_of_strings(
                    training_path / "inputs" / subsys_lammps_data_fn
                )
                subsys_ipi_xyz_fn = it_subsys_nr + ".xyz"
                subsys_ipi_xml_aslist = replace_substring_in_list_of_strings(
                    subsys_ipi_xml_aslist, "_R_XYZ_", subsys_ipi_xyz_fn
                )
                subsys_ipi_json["coord_file"] = subsys_ipi_xyz_fn
                # ### Get the XYZ file from LMP
                subprocess.call(
                    [
                        atomsk_bin,
                        str(training_path / "files" / subsys_lammps_data_fn),
                        "xyz",
                        str(training_path / "files" / it_subsys_nr),
                        "-ow",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
                subsys_ipi_xyz = file_to_list_of_strings(
                    training_path / "files" / subsys_ipi_xyz_fn
                )
                # ### Default time and nb steps
                if with_plumed_smd == 1:
                    subsys_nb_steps = int(with_plumed_smd_nb_steps)
                else:
                    subsys_nb_steps = subsys_init_exp_time_ps / subsys_timestep
                subsys_ipi_xml_aslist = replace_substring_in_list_of_strings(
                    subsys_ipi_xml_aslist, "_R_NB_STEPS_", str(subsys_nb_steps)
                )
                subsys_walltime_approx_s = subsys_init_walltime_h * 3600
                # ### Get the cell and nb of atoms (for config.json and it is needed)
                subsys_cell, subsys_nb_atm = get_cell_nbatoms_from_lmp(
                    subsys_lammps_data
                )
                subsys_ipi_xml_aslist = replace_substring_in_list_of_strings(
                    subsys_ipi_xml_aslist, "_R_CELL_", str(subsys_cell)
                )
                # ### Get the type_map from config (key added after first training)
                for it_zzz, zzz in enumerate(config_json["type_map"]):
                    subsys_ipi_json["atom_type"][str(zzz)] = it_zzz

        print(
            subsys_timestep,
            subsys_temp,
            subsys_init_exp_time_ps,
            subsys_walltime_approx_s,
        )
        print(with_plumed, with_plumed_smd, with_plumed_smd_nb_steps)

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "exploration",
            "preparation",
            Path(sys.argv[1]),
            fake_cluster=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
