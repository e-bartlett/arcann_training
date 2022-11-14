## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## Either shortcut (machine_file.json) or Project name / allocation / arch
# user_spec = "v100"
# user_spec = ["nvs","v100","v100"]
# slurm_email: str = ""
## These are the default
# temperature_K: list = [298.15, 298.15] #float #LAMMPS,  #i-PI is from XML
# timestep_ps: list = [0.0005, 0.0005] #float #LAMMPS
# timestep_ps: list = [0.00025, 0.00025] #float #i-PI
# nb_traj: int = 2
# disturbed_start: list = [False, False] #bool
# Default is every 0.01*nb_steps
# print_every_x_steps: list = [200, 200]
# max_exploration_time_ps: list = [400, 400] #int / LAMMPS
# max_exploration_time_ps: list = [100, 100] #int / i-PI
## nb_steps_exploration / job_walltime_h are auto-calculated (local subsys)
# exploration_time_ps: list = [XX, XX] #int
# job_walltime_h: list = [XX, XX] #int
## Init only:
# initial_exploration_time_ps = [10, 10] #int
# init_job_walltime_h: list = [1, 1] #int / LAMMPS
# init_job_walltime_h: list = [10, 10] #int / LAMMPS

###################################### No change past here
from re import I
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")

import subprocess
import numpy as np
import random

training_iterative_apath = Path("..").resolve()
### Check if the deepmd_iterative_apath is defined
deepmd_iterative_apath_error = 1
if "deepmd_iterative_apath" in globals():
    if (Path(deepmd_iterative_apath)/"tools"/"common_functions.py").is_file():
        deepmd_iterative_apath = Path(deepmd_iterative_apath)
        deepmd_iterative_apath_error = 0
elif (Path().home()/"deepmd_iterative_py"/"tools"/"common_functions.py").is_file():
    deepmd_iterative_apath = Path().home()/"deepmd_iterative_py"
    deepmd_iterative_apath_error = 0
elif (training_iterative_apath/"control"/"path").is_file():
    deepmd_iterative_apath = Path((training_iterative_apath/"control"/"path").read_text())
    if (deepmd_iterative_apath/"tools"/"common_functions.py").is_file():
        deepmd_iterative_apath_error = 0
if deepmd_iterative_apath_error == 1:
    logging.critical("Can\"t find common_functions.py in usual places:")
    logging.critical("deepmd_iterative_apath variable or ~/deepmd_iterative_py or in the path file in control")
    logging.critical("Aborting...")
    sys.exit(1)
sys.path.insert(0, str(deepmd_iterative_apath/"tools"))
del deepmd_iterative_apath_error
import common_functions as cf

slurm_email = "" if "slurm_email" not in globals() else slurm_email

### Read what is needed (json files)
control_apath = training_iterative_apath/"control"
jobs_apath = deepmd_iterative_apath/"jobs"/"exploration"
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)
config_json = cf.json_read((control_apath/"config.json"),True,True)
exploration_json = cf.json_read((control_apath/("exploration_"+current_iteration_zfill+".json")),False,True)

previous_iteration_zfill = str(current_iteration-1).zfill(3)
prevtraining_json = cf.json_read((control_apath/("training_"+previous_iteration_zfill+".json")),True,True)
if int(previous_iteration_zfill) > 0:
    prevexploration_json = cf.json_read((control_apath/("exploration_"+previous_iteration_zfill+".json")),True,True)

exploration_json["deepmd_model_version"] = prevtraining_json["deepmd_model_version"]
exploration_json["nb_nnp"] = config_json["nb_nnp"]
exploration_json["nb_traj"] = 2 if "nb_traj" not in globals() else nb_traj
exploration_json["exploration_type"] = config_json["exploration_type"]
exploration_type = exploration_json["exploration_type"]

### Checks
if not prevtraining_json["is_frozen"]:
    logging.critical("Lock found. Previous NNPs aren\'t frozen")
    logging.critical("Aborting...")
    sys.exit(1)

### Read cluster info
if user_spec in globals():
    cluster, cluster_spec, cluster_error = cf.clusterize(deepmd_iterative_apath,training_iterative_apath,step="exploration",user_keyword=user_spec)
else:
    cluster, cluster_spec, cluster_error = cf.clusterize(deepmd_iterative_apath,training_iterative_apath,step="exploration")
if cluster_error != 0:
    ###FIXME Better errors
    sys.exit(1)

exploration_json["cluster"] = cluster
exploration_json["project_name"] = cluster_spec["project_name"]
exploration_json["allocation_name"] = cluster_spec["allocation_name"]
exploration_json["arch_name"] = cluster_spec["arch_name"]
exploration_json["arch_type"] = cluster_spec["arch_type"]

cf.check_file(jobs_apath/("job_deepmd_"+exploration_type+"_"+cluster_spec["arch_type"]+"_"+cluster+".sh"),True,True,"No SLURM file present for the exploration step on this cluster.")
slurm_file_master = cf.read_file(jobs_apath/("job_deepmd_"+exploration_type+"_"+cluster_spec["arch_type"]+"_"+cluster+".sh"))
del jobs_apath

### Preparation of the exploration
exploration_json["subsys_nr"]={}
for it0_subsys_nr,it_subsys_nr in enumerate(config_json["subsys_nr"]):
    random.seed()
    exploration_json["subsys_nr"][it_subsys_nr]={}

    subsys_temp = config_json["subsys_nr"][it_subsys_nr]["temperature_K"] if "temperature_K" not in globals() else temperature_K[it0_subsys_nr]
    subsys_timestep = config_json["subsys_nr"][it_subsys_nr]["timestep_ps"] if "timestep_ps" not in globals() else timestep_ps[it0_subsys_nr]

    exploration_json["subsys_nr"][it_subsys_nr]["temperature_K"] = subsys_temp
    exploration_json["subsys_nr"][it_subsys_nr]["timestep_ps"] = subsys_timestep

    if exploration_type == "lammps":

        subsys_exploration_input = cf.read_file(training_iterative_apath/"inputs"/(it_subsys_nr+".in"))
        subsys_exploration_input = cf.replace_in_list(subsys_exploration_input,"_R_TEMPERATURE_",str(subsys_temp))
        subsys_exploration_input = cf.replace_in_list(subsys_exploration_input,"_R_TIMESTEP_",str(subsys_timestep))

        if current_iteration == 1:
            subsys_nb_steps = 20000 if "nb_steps_initial" not in globals() else nb_steps_initial[it0_subsys_nr]
            subsys_exploration_input = cf.replace_in_list(subsys_exploration_input,"_R_NUMBER_OF_STEPS_",str(subsys_nb_steps))
            subsys_lammps_data_fn = it_subsys_nr+".lmp"
            subsys_exploration_input = cf.replace_in_list(subsys_exploration_input,"_R_DATA_FILE_",subsys_lammps_data_fn)
            subsys_lammps_data = cf.read_file(training_iterative_apath/"inputs"/subsys_lammps_data_fn)
            subsys_walltime_approx_s = 36000 if "init_job_walltime_h" not in globals() else init_job_walltime_h[it0_subsys_nr]*3600

            ### Get cell and number of atoms
            dim_string = ["xlo xhi", "ylo yhi", "zlo zhi"]
            subsys_cell=[]
            for n,string in enumerate(dim_string):
                temp = [zzz for zzz in subsys_lammps_data if string in zzz]
                temp = cf.replace_in_list(temp,"\n","")
                temp = [zzz for zzz in temp[0].split(" ") if zzz]
                subsys_cell.append(float(temp[1]) - float(temp[0]))
            del n, string
            temp = [zzz for zzz in subsys_lammps_data if "atoms" in zzz]
            temp = cf.replace_in_list(temp,"\n","")
            temp = [zzz for zzz in temp[0].split(" ") if zzz]
            subsys_nb_atm = int(temp[0])
            del temp, dim_string

    elif exploration_type == 'i-PI':
        ### #12
        True

    ### Starting structures (after first iteration)
    if current_iteration > 1:

        if exploration_type == "lammps":
            starting_point_list_path = [zzz for zzz in (training_iterative_apath/"starting_structures").glob(previous_iteration_zfill+"_"+it_subsys_nr+"_*.lmp")]
        elif exploration_type == 'i-PI':
            ### #12
            starting_point_list_path = [zzz for zzz in (training_iterative_apath/"starting_structures").glob(previous_iteration_zfill+"_"+it_subsys_nr+"_*.xyz")]
        starting_point_list_all = [str(zzz).split("/")[-1] for zzz in starting_point_list_path]
        starting_point_list = [zzz for zzz in starting_point_list_all if "disturbed" not in zzz]
        starting_point_list_disturbed = [zzz for zzz in starting_point_list_all if zzz not in starting_point_list]
        starting_point_list_disturbed_bckp = starting_point_list_disturbed.copy()
        starting_point_list_bckp = starting_point_list.copy()

        if "disturbed_start" not in globals():
            if (
                current_iteration > 1
                and prevexploration_json["subsys_nr"][it_subsys_nr]["disturbed_start"]
                and prevexploration_json["subsys_nr"][it_subsys_nr]["disturbed_min"]
                ):
                starting_point_list = starting_point_list_disturbed.copy()
                starting_point_list_bckp = starting_point_list_disturbed_bckp.copy()
                exploration_json["subsys_nr"][it_subsys_nr]["disturbed_start"] = True
            else:
                exploration_json["subsys_nr"][it_subsys_nr]["disturbed_start"] = False
        elif disturbed_start[it0_subsys_nr]:
            starting_point_list = starting_point_list_disturbed.copy()
            starting_point_list_bckp = starting_point_list_disturbed_bckp.copy()
            exploration_json["subsys_nr"][it_subsys_nr]["disturbed_start"] = disturbed_start[it0_subsys_nr]
        else:
            exploration_json["subsys_nr"][it_subsys_nr]["disturbed_start"] = False
        del starting_point_list_path, starting_point_list_all

    for it_nnp in range(1, config_json["nb_nnp"] + 1 ):
        for it_number in range(1, exploration_json["nb_traj"] + 1):

            local_apath = Path(".").resolve()/str(it_subsys_nr)/str(it_nnp)/(str(it_number).zfill(5))
            local_apath.mkdir(exist_ok=True,parents=True)

            ### Get the model list
            list_nnp = [zzz for zzz in range(1, config_json["nb_nnp"] + 1)]
            reorder_nnp_list = list_nnp[list_nnp.index(it_nnp):] + list_nnp[:list_nnp.index(it_nnp)]
            compress_str = "_compressed" if prevtraining_json["is_compressed"] else ""
            models_list=["graph_"+str(f)+"_"+previous_iteration_zfill+compress_str+".pb" for f in reorder_nnp_list]
            for it_sub_nnp in range(1, config_json["nb_nnp"] + 1 ):
                nnp_apath = (training_iterative_apath/"NNP"/("graph_"+str(it_sub_nnp)+"_"+previous_iteration_zfill+compress_str+".pb")).resolve()
                subprocess.call(["ln","-s", str(nnp_apath), str(local_apath)])
            models_list=" ".join(models_list)
            del list_nnp, it_sub_nnp, nnp_apath, compress_str, reorder_nnp_list

            ### LAMMPS
            if exploration_type == "lammps":
                exploration_input = subsys_exploration_input.copy()
                RAND = random.randrange(0,1000)
                exploration_input = cf.replace_in_list(exploration_input,"_R_SEED_VEL_",str(it_nnp)+str(RAND)+str(it_number)+previous_iteration_zfill)
                RAND = random.randrange(0,1000)
                exploration_input = cf.replace_in_list(exploration_input,"_R_SEED_THER_",str(it_nnp)+str(RAND)+str(it_number)+previous_iteration_zfill)
                exploration_input = cf.replace_in_list(exploration_input,"_R_DCD_OUT_",str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".dcd")
                exploration_input = cf.replace_in_list(exploration_input,"_R_RESTART_OUT_",str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".restart")
                exploration_input = cf.replace_in_list(exploration_input,"_R_MODELS_LIST_",models_list)
                exploration_input = cf.replace_in_list(exploration_input,"_R_DEVI_OUT_","model_devi_"+str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".out")
                del RAND

                ### Get data files (starting points) and number of steps
                if current_iteration > 1:
                    if len(starting_point_list) == 0:
                        starting_point_list = starting_point_list_bckp.copy()
                    RAND = random.randrange(0,len(starting_point_list))
                    subsys_lammps_data_fn = starting_point_list[RAND]
                    subsys_lammps_data = cf.read_file(training_iterative_apath/"starting_structures"/subsys_lammps_data_fn)
                    exploration_input = cf.replace_in_list(exploration_input,"_R_DATA_FILE_",subsys_lammps_data_fn)

                    ratio_ill_described = (
                        (prevexploration_json["subsys_nr"][it_subsys_nr]["nb_candidates"] + prevexploration_json["subsys_nr"][it_subsys_nr]["nb_rejected"])
                        / prevexploration_json["subsys_nr"][it_subsys_nr]["nb_total"]
                        )
                    subsys_nb_steps = prevexploration_json["subsys_nr"][it_subsys_nr]["nb_steps"]
                    if ( ratio_ill_described ) < 0.10:
                        subsys_nb_steps = subsys_nb_steps * 4
                    elif ( ratio_ill_described ) < 0.20:
                        subsys_nb_steps = subsys_nb_steps * 2

                    subsys_max_exploration_time_ps = 400 if "max_exploration_time_ps" not in globals() else max_exploration_time_ps[it0_subsys_nr]
                    if subsys_nb_steps > subsys_max_exploration_time_ps/subsys_timestep:
                        subsys_nb_steps = int(subsys_max_exploration_time_ps/subsys_nb_steps)

                    subsys_nb_steps = subsys_nb_steps if "nb_steps_exploration" not in globals() else nb_steps_exploration[it0_subsys_nr]
                    exploration_input = cf.replace_in_list(exploration_input,"_R_NUMBER_OF_STEPS_",str(subsys_nb_steps))

                    subsys_walltime_approx_s = ( prevexploration_json["subsys_nr"][it_subsys_nr]["s_per_step"] * subsys_nb_steps )
                    subsys_walltime_approx_s = subsys_walltime_approx_s * 1.10
                    subsys_walltime_approx_s = subsys_walltime_approx_s if "job_walltime_h" not in globals() else int(job_walltime_h[it0_subsys_nr] * 3600 )

                    del starting_point_list[RAND]
                    del RAND

                ### Write DATA file
                cf.write_file(local_apath/subsys_lammps_data_fn,subsys_lammps_data)

                ### Get print freq
                it_print_every_x_steps = int(subsys_nb_steps*0.01) if "print_every_x_steps" not in globals() else print_every_x_steps[it0_subsys_nr]
                exploration_input = cf.replace_in_list(exploration_input,"_R_print_every_x_steps_",str(it_print_every_x_steps))

                ### Plumed files
                if any("plumed" in f for f in exploration_input):
                    list_plumed_files=[x for x in (training_iterative_apath/"inputs").glob("*plumed*_"+it_subsys_nr+".dat")]
                    if len(list_plumed_files) == 0 :
                        logging.critical("Plumed in LAMMPS input but no plumed files")
                        logging.critical("Aborting...")
                        sys.exit(1)
                    plumed_input={}
                    for it_list_plumed_files in list_plumed_files:
                        plumed_input[it_list_plumed_files.name] = cf.read_file(it_list_plumed_files)
                    exploration_input = cf.replace_in_list(exploration_input,"_R_PLUMED_IN_","plumed_"+str(it_subsys_nr)+".dat")
                    exploration_input = cf.replace_in_list(exploration_input,"_R_PLUMED_OUT_","plumed_"+str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".log")
                    for it_plumed_input in plumed_input:
                        plumed_input[it_plumed_input] = cf.replace_in_list(plumed_input[it_plumed_input],"_R_print_every_x_steps_",str(it_print_every_x_steps))
                        cf.write_file(local_apath/it_plumed_input,plumed_input[it_plumed_input])
                    del list_plumed_files, it_list_plumed_files

                exploration_json["subsys_nr"][it_subsys_nr]["nb_steps"] = subsys_nb_steps
                exploration_json["subsys_nr"][it_subsys_nr]["print_every_x_steps"] = it_print_every_x_steps

                ### Write INPUT file
                cf.write_file(local_apath/(str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".in"),exploration_input)

                slurm_file = slurm_file_master.copy()
                slurm_file = cf.replace_in_list(slurm_file,"_R_DEEPMD_VERSION_",str(exploration_json["deepmd_model_version"]))
                slurm_file = cf.replace_in_list(slurm_file,"_R_INPUT_",str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill)
                slurm_file = cf.replace_in_list(slurm_file,"_R_DATA_FILE_",subsys_lammps_data_fn)

                slurm_file = cf.replace_in_list(slurm_file,"_R_PROJECT_",cluster_spec["project_name"])
                slurm_file = cf.replace_in_list(slurm_file,"_R_ALLOC_",cluster_spec["allocation_name"])
                slurm_file = cf.delete_in_list(slurm_file,"_R_PARTITON_") if cluster_spec["partition"] is None else cf.replace_in_list(slurm_file,"_R_PARTITION_",cluster_spec["partition"])
                slurm_file = cf.delete_in_list(slurm_file,"_R_SUBPARTITION_") if cluster_spec["subpartition"] is None else cf.replace_in_list(slurm_file,"_R_SUBPARTITION_",cluster_spec["subpartition"])
                max_qos_time = 0
                for it_qos in cluster_spec["qos"]:
                    if cluster_spec["qos"][it_qos] >= subsys_walltime_approx_s:
                        slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_",it_qos)
                        qos_ok = True
                    else:
                        max_qos = it_qos if cluster_spec["qos"][it_qos] > max_qos_time else max_qos
                        qos_ok = False
                del it_qos
                if not qos_ok:
                    logging.warning("Approximate wall time superior than the maximun time allowed by the QoS")
                    logging.warning("Settign the maximum QoS time as walltime")
                    slurm_file = cf.replace_in_list(slurm_file,"_R_WALLTIME_",str(max_qos_time))
                else:
                    slurm_file = cf.replace_in_list(slurm_file,"_R_WALLTIME_",str(subsys_walltime_approx_s))

                if slurm_email != "":
                    slurm_file = cf.replace_in_list(slurm_file,"_R_EMAIL_",slurm_email)
                else:
                    slurm_file = cf.delete_in_list(slurm_file,"_R_EMAIL_")
                    slurm_file = cf.delete_in_list(slurm_file,"mail")

                ### Add plumed files
                if any("plumed" in f for f in exploration_input):
                    for n,it_plumed_input in enumerate(plumed_input):
                        if n == 0:
                            slurm = cf.replace_in_list(slurm,"_R_PLUMED_FILES_LIST_",it_plumed_input)
                        else:
                            slurm = cf.replace_in_list(slurm,prev_plumed,prev_plumed+"\" \""+it_plumed_input)
                        prev_plumed = it_plumed_input
                    del n, it_plumed_input, plumed_input, prev_plumed
                else:
                    slurm = cf.replace_in_list(slurm," \"_R_PLUMED_FILES_LIST_\"","")

                models_list_job = models_list.replace(" ","\" \"")
                slurm = cf.replace_in_list(slurm, "_R_MODELS_LIST_", models_list_job)

                cf.write_file(local_apath/("job_deepmd_"+exploration_type+"_"+cluster_spec["arch_type"]+"_"+cluster+".sh"),slurm)

                del exploration_input, slurm, models_list_job

            elif exploration_type == "i-PI":
                ### #12
                True

            del local_apath, models_list

        del it_number

    del it_nnp

    config_json["subsys_nr"][it_subsys_nr]["cell"] = subsys_cell
    config_json["subsys_nr"][it_subsys_nr]["nb_atm"] = subsys_nb_atm

    del subsys_temp, subsys_cell, subsys_nb_atm, subsys_nb_steps, subsys_exploration_input
    del subsys_lammps_data, subsys_timestep, subsys_lammps_data_fn, subsys_walltime_approx_s, it_print_every_x_steps

del it0_subsys_nr, it_subsys_nr, slurm_file_master

exploration_json["is_locked"] = True
exploration_json["is_launched"] = False
exploration_json["is_checked"] = False
if exploration_type == 'i-PI':
    ### #12
    exploration_json["is_unbeaded"] = False
    exploration_json["is_reruned"] = False
    exploration_json["is_rechecked"] = False
exploration_json["is_deviated"] = False
exploration_json["is_extracted"] = False
del exploration_type

## Dump the config/training
cf.json_dump(config_json,(control_apath/"config.json"),True)
cf.json_dump(exploration_json,(control_apath/("exploration_"+current_iteration_zfill+".json")),True)

logging.info("Exploration: Prep phase is a success!")

### Cleaning
del config_json, training_iterative_apath, control_apath
del current_iteration, current_iteration_zfill
del exploration_json
del cluster, cluster_spec
del deepmd_iterative_apath
del slurm_email

del previous_iteration_zfill
del prevtraining_json

del sys, Path, logging, cf
del subprocess, np, random
import gc; gc.collect(); del gc
exit()