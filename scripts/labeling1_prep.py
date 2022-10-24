## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## Project name / allocation / arch (nvs/v100/a100 or gen7156/rome/cpu)
# project_name: str = "nvs"
# allocation_name: str = "v100"
# arch_name: str = "v100"
# slurm_email: str = ""
# cp2k_1_walltime_h: list = [0.5, 0.5]
# cp2k_2_walltime_h: list = [1.0, 1.0]
# nb_NODES: list = [1, 1]
# nb_MPI_per_NODE: list = [10, 10]
# nb_OPENMP_per_MPI: list = [1, 1]

###################################### No change past here
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")

import subprocess

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
    logging.critical("Can\'t find common_functions.py in usual places:")
    logging.critical("deepmd_iterative_apath variable or ~/deepmd_iterative_py or in the path file in control")
    logging.critical("Aborting...")
    sys.exit(1)
sys.path.insert(0, str(deepmd_iterative_apath/"tools"))
del deepmd_iterative_apath_error
import common_functions as cf

slurm_email = "" if "slurm_email" not in globals() else slurm_email

### Read what is needed (json files)
control_apath = training_iterative_apath/"control"
jobs_apath = deepmd_iterative_apath/"jobs"/"labeling"
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)
config_json = cf.json_read((control_apath/"config.json"),True,True)
exploration_json = cf.json_read((control_apath/("exploration_"+current_iteration_zfill+".json")),True,True)
labeling_json = cf.json_read((control_apath/("labeling_"+current_iteration_zfill+".json")),False,True)

### Checks
if not exploration_json["is_extracted"]:
    logging.critical("Lock found. Run/Check first: exploration5_extract.py")
    logging.critical("Aborting...")
    sys.exit(1)

### #35
cluster = cf.check_cluster()

if cluster != "ir" and cluster != "jz":
    logging.critical("Unsupported cluster.")
    logging.critical("Aborting...")
    sys.exit(1)

### #35
labeling_json["cluster"] = cluster
labeling_json["project_name"] = project_name if "project_name" in globals() else "nvs"
labeling_json["allocation_name"] = allocation_name if "allocation_name" in globals() else "cpu"
labeling_json["arch_name"] = arch_name if "arch_name" in globals() else "cpu"
project_name = labeling_json["project_name"]
allocation_name = labeling_json["allocation_name"]
arch_name = labeling_json["arch_name"]
if arch_name == "cpu":
    arch_type ="cpu"

### #35
cf.check_file(jobs_apath/("job_labeling_XXXXX_"+arch_type +"_"+cluster+".sh"),True,True,"No SLURM file present for the exploration step on this cluster.")
cf.check_file(jobs_apath/("job_labeling_array_"+arch_type +"_"+cluster+".sh"),True,True,"No SLURM Array present for the exploration step on this cluster.")

### Preparation of the labeling
slurm_file_master = cf.read_file(jobs_apath/("job_labeling_XXXXX_"+arch_type+"_"+cluster+".sh"))
slurm_file_master = cf.replace_in_list(slurm_file_master,"_R_PROJECT_",project_name)
slurm_file_master = cf.replace_in_list(slurm_file_master,"_R_ALLOC_",allocation_name)
slurm_file_array_master = cf.read_file(jobs_apath/("job_labeling_array_"+arch_type+"_"+cluster+".sh"))
slurm_file_array_master = cf.replace_in_list(slurm_file_array_master,"_R_PROJECT_",project_name)
slurm_file_array_master = cf.replace_in_list(slurm_file_array_master,"_R_ALLOC_",allocation_name)
if slurm_email != "":
    if cluster == "jz":
        slurm_file_array_master = cf.replace_in_list(slurm_file_array_master,"##SBATCH --mail-type","#SBATCH --mail-type")
        slurm_file_array_master = cf.replace_in_list(slurm_file_array_master,"##SBATCH --mail-user _R_EMAIL_","#SBATCH --mail-user "+slurm_email)
    elif cluster == "ir":
        slurm_file_array_master = cf.replace_in_list(slurm_file_array_master,"##MSUB -@ _R_EMAIL_","##SUB -@ "+slurm_email)

labeling_json["subsys_nr"] = {}
subsys_list=list(config_json["subsys_nr"].keys())
for it0_subsys_nr, it_subsys_nr in enumerate(subsys_list):

    nb_candidates = int(exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates_kept"])
    nb_candidates_disturbed = int(exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates_kept"]) if exploration_json["subsys_nr"][it_subsys_nr]["disturbed_candidates"] is True else 0
    nb_steps = nb_candidates + nb_candidates_disturbed

    subsys_apath = Path(".").resolve()/str(it_subsys_nr)
    subsys_apath.mkdir(exist_ok=True)

    labeling_json["subsys_nr"][it_subsys_nr] = {}
    labeling_json["subsys_nr"][it_subsys_nr]["cp2k_1_walltime_h"] = 0.5 if "cp2k_1_walltime_h" not in globals() else cp2k_1_walltime_h[it0_subsys_nr]
    labeling_json["subsys_nr"][it_subsys_nr]["cp2k_2_walltime_h"] = 1.0 if "cp2k_1_walltime_h" not in globals() else cp2k_2_walltime_h[it0_subsys_nr]
    labeling_json["subsys_nr"][it_subsys_nr]["nb_NODES"] = 1 if "nb_NODES" not in globals() else nb_NODES[it0_subsys_nr]
    labeling_json["subsys_nr"][it_subsys_nr]["nb_MPI_per_NODE"] = 10 if "nb_MPI_per_NODE" not in globals() else nb_MPI_per_NODE[it0_subsys_nr]
    labeling_json["subsys_nr"][it_subsys_nr]["nb_OPENMP_per_MPI"] = 1 if "nb_OPENMP_per_MPI" not in globals() else nb_OPENMP_per_MPI[it0_subsys_nr]

    slurm_file_subsys = cf.replace_in_list(slurm_file_master,"_R_nb_NODES_",str(labeling_json["subsys_nr"][it_subsys_nr]["nb_NODES"]))
    slurm_file_subsys = cf.replace_in_list(slurm_file_subsys,"_R_nb_OPENMP_per_MPI_",str(labeling_json["subsys_nr"][it_subsys_nr]["nb_OPENMP_per_MPI"]))
    if cluster == "jz":
        slurm_file_subsys = cf.replace_in_list(slurm_file_subsys,"_R_nb_MPI_per_NODE_",str(labeling_json["subsys_nr"][it_subsys_nr]["nb_MPI_per_NODE"]))
    elif cluster == "ir":
        slurm_file_subsys = cf.replace_in_list(slurm_file_subsys,"_R_nb_MPI_",str(labeling_json["subsys_nr"][it_subsys_nr]["nb_MPI_per_NODE"] * labeling_json["subsys_nr"][it_subsys_nr]["nb_NODES"] ))

    slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_master,"_R_nb_NODES_",str(labeling_json["subsys_nr"][it_subsys_nr]["nb_NODES"]))
    slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_nb_OPENMP_per_MPI_",str(labeling_json["subsys_nr"][it_subsys_nr]["nb_OPENMP_per_MPI"]))
    if cluster == "jz":
            slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_nb_MPI_per_NODE_",str(labeling_json["subsys_nr"][it_subsys_nr]["nb_MPI_per_NODE"]))
    elif cluster == "ir":
         slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_nb_MPI_",str(labeling_json["subsys_nr"][it_subsys_nr]["nb_MPI_per_NODE"] * labeling_json["subsys_nr"][it_subsys_nr]["nb_NODES"] ))
    slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_CP2K_JOBNAME_","CP2K_"+it_subsys_nr+"_"+current_iteration_zfill)

    slurm_walltime_s = (labeling_json["subsys_nr"][it_subsys_nr]["cp2k_1_walltime_h"] + labeling_json["subsys_nr"][it_subsys_nr]["cp2k_2_walltime_h"]) * 3600
    slurm_walltime_s = int(slurm_walltime_s * 1.1)

    if cluster == "jz":
        slurm_file_subsys = cf.replace_in_list(slurm_file_subsys,"_R_WALLTIME_",cf.seconds_to_walltime(slurm_walltime_s))
        slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_WALLTIME_",cf.seconds_to_walltime(slurm_walltime_s))
        slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_ARRAYCOUNT_",str(nb_steps))
        cf.write_file(subsys_apath/("job_labeling_array_"+arch_type+"_"+cluster+".sh"),slurm_file_array_subsys)
        slurm_file_subsys = cf.replace_in_list(slurm_file_subsys,"_R_WALLTIME_",cf.seconds_to_walltime(slurm_walltime_s))

    elif cluster == "ir":
        slurm_file_subsys = cf.replace_in_list(slurm_file_subsys,"_R_WALLTIME_",str(slurm_walltime_s))
        slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_WALLTIME_",str(slurm_walltime_s))
        if nb_steps <= 1000:
            if nb_steps <= 250:
                slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_ARRAY_START_",str(1))
                slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_ARRAY_END_",str(nb_steps))
                cf.write_file(subsys_apath/("job_labeling_array_"+arch_type+"_"+cluster+"_0.sh"),slurm_file_array_subsys)
            else:
                slurm_file_array_subsys_dict={}
                quotient = nb_steps // 250
                remainder = nb_steps % 250
                slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_NEW_START_","0")
                for i in range(0,quotient+1):
                    if i < quotient:
                        slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys,"_R_ARRAY_START_",str(250*i + 1))
                        slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(i)],"_R_ARRAY_END_",str(250 * (i+1)))
                        slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(i)],"_R_LAUNCHNEXT_","1")
                        slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(i)],"_R_NEXT_JOB_FILE_",str(i+1))
                        slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(i)],"_R_CD_WHERE_","${SLURM_SUBMIT_DIR}")
                        cf.write_file(subsys_apath/("job_labeling_array_"+arch_type+"_"+cluster+"_"+str(i)+".sh"),slurm_file_array_subsys_dict[str(i)])
                    else:
                        slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys,"_R_ARRAY_START_",str(250*i + 1))
                        slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(i)],"_R_ARRAY_END_",str(250*i + remainder ))
                        if it0_subsys_nr != len(config_json["subsys_nr"]) - 1:
                            slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(i)],"_R_LAUNCHNEXT_","1")
                            slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(i)],"_R_NEXT_JOB_FILE_","0")
                            slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(i)],"_R_CD_WHERE_","${SLURM_SUBMIT_DIR}/../"+subsys_list[it0_subsys_nr+1])
                        else:
                            True
                        cf.write_file(subsys_apath/("job_labeling_array_"+arch_type+"_"+cluster+"_"+str(i)+".sh"),slurm_file_array_subsys_dict[str(i)])
                del quotient, remainder, i
                del slurm_file_array_subsys_dict
        else:
            slurm_file_array_subsys_dict={}
            quotient = nb_steps // 1000
            remainder = nb_steps % 1000
            m = 0
            for i in range(0,quotient+1):
                if i < quotient:
                    for j in range(0,4):
                        slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys ,"_R_NEW_START_",str(i*1000))
                        slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_ARRAY_START_",str(250*j + 1))
                        slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_ARRAY_END_",str(250 * (j+1)))
                        slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_LAUNCHNEXT_","1")
                        slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_NEXT_JOB_FILE_",str(m+1))
                        slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_CD_WHERE_","${SLURM_SUBMIT_DIR}")
                        cf.write_file(subsys_apath/("job_labeling_array_"+arch_type+"_"+cluster+"_"+str(m)+".sh"),slurm_file_array_subsys_dict[str(m)])
                        m = m + 1
                else:
                    quotient2 = remainder // 250
                    remainder2 = remainder % 250
                    for j in range(0,quotient2+1):
                        if j < quotient2:
                            slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys ,"_R_NEW_START_",str(i*1000))
                            slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_ARRAY_START_",str(250*j + 1))
                            slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_ARRAY_END_",str(250 * (j+1)))
                            slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_LAUNCHNEXT_","1")
                            slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_NEXT_JOB_FILE_",str(m+1))
                            slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_CD_WHERE_","${SLURM_SUBMIT_DIR}")
                            cf.write_file(subsys_apath/("job_labeling_array_"+arch_type+"_"+cluster+"_"+str(m)+".sh"),slurm_file_array_subsys_dict[str(m)])
                            m = m + 1
                        else:
                            slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys ,"_R_NEW_START_",str(i*1000))
                            slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_ARRAY_START_",str(250*j + 1))
                            slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_ARRAY_END_",str(250*j + remainder2))
                            if it0_subsys_nr != len(config_json["subsys_nr"]) - 1:
                                slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_LAUNCHNEXT_","1")
                                slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_NEXT_JOB_FILE_","0")
                                ### #TODO This is not Path friendly
                                slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_CD_WHERE_","${SLURM_SUBMIT_DIR}/../"+subsys_list[it0_subsys_nr+1])
                            else:
                                True
                            cf.write_file(subsys_apath/("job_labeling_array_"+arch_type+"_"+cluster+"_"+str(m)+".sh"),slurm_file_array_subsys_dict[str(m)])
                            m = m + 1
                    del quotient2, remainder2, j
            del m, quotient, remainder, m, i
            del slurm_file_array_subsys_dict

    xyz_file = training_iterative_apath/(current_iteration_zfill+"-exploration")/it_subsys_nr/("candidates_"+str(it_subsys_nr)+"_"+current_iteration_zfill+".xyz")
    xyz_file_disturbed = training_iterative_apath/(current_iteration_zfill+"-exploration")/it_subsys_nr/("candidates_"+str(it_subsys_nr)+"_"+current_iteration_zfill+"_disturbed.xyz")

    cp2k_input_1 = cf.read_file(training_iterative_apath/"inputs"/("1_"+str(it_subsys_nr)+"_labeling_XXXXX_"+cluster+".inp"))
    cp2k_input_2 = cf.read_file(training_iterative_apath/"inputs"/("2_"+str(it_subsys_nr)+"_labeling_XXXXX_"+cluster+".inp"))

    cp2k_input_1 = cf.replace_in_list(cp2k_input_1,"_R_CELL_"," ".join([str(zzz) for zzz in config_json["subsys_nr"][it_subsys_nr]["cell"]]))
    cp2k_input_2 = cf.replace_in_list(cp2k_input_2,"_R_CELL_"," ".join([str(zzz) for zzz in config_json["subsys_nr"][it_subsys_nr]["cell"]]))
    cp2k_input_1 = cf.replace_in_list(cp2k_input_1,"_R_WALLTIME_",str(round(labeling_json["subsys_nr"][it_subsys_nr]["cp2k_1_walltime_h"],2) * 3600))
    cp2k_input_2 = cf.replace_in_list(cp2k_input_2,"_R_WALLTIME_",str(round(labeling_json["subsys_nr"][it_subsys_nr]["cp2k_2_walltime_h"],2) * 3600))

    ###
    n_atom, step_atoms, step_coordinates, blank = cf.import_xyz(xyz_file)

    for step_iter in range(1,step_atoms.shape[0]+1):
        step_iter_str = str(step_iter).zfill(5)
        (subsys_apath/step_iter_str).mkdir(exist_ok=True)
        cf.check_dir((subsys_apath/step_iter_str),True)

        cp2k_input_t1 = cf.replace_in_list(cp2k_input_1,"XXXXX",step_iter_str)
        cp2k_input_t2 = cf.replace_in_list(cp2k_input_2,"XXXXX",step_iter_str)

        cf.write_file(subsys_apath/step_iter_str/("1_labeling_"+step_iter_str+".inp"),cp2k_input_t1)
        cf.write_file(subsys_apath/step_iter_str/("2_labeling_"+step_iter_str+".inp"),cp2k_input_t2)

        slurm_file = cf.replace_in_list(slurm_file_subsys,"XXXXX",step_iter_str)
        slurm_file = cf.replace_in_list(slurm_file,"_R_CP2K_JOBNAME_","CP2K_"+it_subsys_nr+"_"+current_iteration_zfill)

        cf.write_file(subsys_apath/step_iter_str/("job_labeling_"+step_iter_str+"_"+arch_type+"_"+cluster+".sh"),slurm_file)
        cf.write_xyz_from_index(subsys_apath/step_iter_str/("labeling_"+step_iter_str+".xyz"),step_iter-1,n_atom,step_atoms,step_coordinates,blank)
        end_step = step_iter

    del n_atom, step_atoms, step_coordinates, blank, step_iter, step_iter_str
    labeling_json["subsys_nr"][it_subsys_nr]["candidates"] = end_step

    if xyz_file_disturbed.is_file():
        ###
        n_atom, step_atoms, step_coordinates, blank = cf.import_xyz(xyz_file_disturbed)

        for d_step_iter in range(end_step+1,step_atoms.shape[0]+end_step+1):
            d_step_iter_str = str(d_step_iter).zfill(5)
            (subsys_apath/d_step_iter_str).mkdir(exist_ok=True)
            cf.check_dir((subsys_apath/d_step_iter_str),True)

            cf.write_file(subsys_apath/d_step_iter_str/("1_labeling_"+d_step_iter_str+".inp"),cp2k_input_t1)
            cf.write_file(subsys_apath/d_step_iter_str/("2_labeling_"+d_step_iter_str+".inp"),cp2k_input_t2)

            slurm_file=cf.replace_in_list(slurm_file_subsys,"XXXXX",d_step_iter_str)
            slurm_file=cf.replace_in_list(slurm_file,"_R_CP2K_JOBNAME_","CP2K_"+it_subsys_nr+"_"+current_iteration_zfill)

            cf.write_file(subsys_apath/d_step_iter_str/("job_labeling_"+d_step_iter_str+"_"+arch_type+"_"+cluster+".sh"),slurm_file)
            cf.write_xyz_from_index(subsys_apath/d_step_iter_str/("labeling_"+d_step_iter_str+".xyz"),d_step_iter-end_step-1,n_atom,step_atoms,step_coordinates,blank)
            end_step_disturbed=d_step_iter

        labeling_json["subsys_nr"][it_subsys_nr]["candidates_disturbed"] = d_step_iter-end_step
        del n_atom, step_atoms, step_coordinates, blank, d_step_iter, d_step_iter_str, end_step, end_step_disturbed
    else:
        labeling_json["subsys_nr"][it_subsys_nr]["candidates_disturbed"] = 0

    del subsys_apath

del it0_subsys_nr, it_subsys_nr, subsys_list
del nb_candidates, nb_candidates_disturbed, nb_steps
del xyz_file, xyz_file_disturbed
del cp2k_input_1, cp2k_input_2, cp2k_input_t1, cp2k_input_t2
del slurm_file_master, slurm_file_subsys, slurm_file
del slurm_file_array_master, slurm_file_array_subsys
del slurm_walltime_s

labeling_json["is_locked"] = True
labeling_json["is_launched"] = False
labeling_json["is_checked"] = False
labeling_json["is_extracted"] = False

cf.json_dump(labeling_json,(control_apath/("labeling_"+current_iteration_zfill+".json")),True)

logging.info("Labeling: Prep phase is a success!")

### Cleaning
del config_json, training_iterative_apath, control_apath, jobs_apath
del current_iteration, current_iteration_zfill
del labeling_json
del exploration_json
del cluster, arch_type
del project_name, allocation_name, arch_name
del deepmd_iterative_apath
del slurm_email

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()