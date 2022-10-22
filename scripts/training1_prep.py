## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## Project name / allocation / arch (nvs/v100/a100 or gen7156/rome/cpu)
# project_name: str = "nvs"
# allocation_name: str = "v100"
# arch_name: str = "v100"
# slurm_email: str = ""
## Training Parameters (Here are the default defaults)
# use_initial_datasets: bool = True
# use_extra_datasets: bool = False
# start_lr: float = 0.001
# stop_lr: float = 1e-06
# decay_rate: float = 0.90
# decay_steps: int = 5000
# numb_steps: int = 400000
# numb_test: int = 0
# deepmd_model_version: float = 2.1
# deepmd_model_type_descriptor: str = "se_e2_a"
## Guess for initial training walltime
# initial_seconds_per_1000steps: float = 90

###################################### No change past here
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
    if (Path(deepmd_iterative_apath)/"scripts"/"common_functions.py").is_file():
        deepmd_iterative_apath = Path(deepmd_iterative_apath)
        deepmd_iterative_apath_error = 0
elif (Path().home()/"deepmd_iterative_py"/"scripts"/"common_functions.py").is_file():
    deepmd_iterative_apath = Path().home()/"deepmd_iterative_py"
    deepmd_iterative_apath_error = 0
elif (training_iterative_apath/"control"/"path").is_file():
    deepmd_iterative_apath = Path((training_iterative_apath/"control"/"path").read_text())
    if (deepmd_iterative_apath/"scripts"/"common_functions.py").is_file():
        deepmd_iterative_apath_error = 0
if deepmd_iterative_apath_error == 1:
    logging.critical("Can\'t find common_functions.py in usual places:")
    logging.critical("deepmd_iterative_apath variable or ~/deepmd_iterative_py or in the path file in control")
    logging.critical("Aborting...")
    sys.exit(1)
sys.path.insert(0, str(deepmd_iterative_apath/"scripts"))
del deepmd_iterative_apath_error
import common_functions as cf

### Read the config file
control_apath = training_iterative_apath/"control"
config_json = cf.json_read((control_apath/"config.json"),True,True)
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)
jobs_apath = deepmd_iterative_apath/"jobs"/"training"

### #35
if "arch_name" in globals() and ( arch_name != "v100" or arch_name != "a100" ):
    logging.critical("Invalid arch_name: "+ arch_name)
    logging.critical("Aborting...")
    sys.exit(1)

if current_iteration > 0:
    labeling_json = cf.json_read((control_apath/("labeling_"+current_iteration_zfill+".json")),True,True)
    if not labeling_json["is_extracted"]:
        logging.critical("Lock found. Run/Check first: labeling4_extract.py")
        logging.critical("Aborting...")
        sys.exit(1)

### Check the cluster name
cluster = cf.check_cluster()

### Get/Create training parameters
training_json = cf.json_read((control_apath/("training_"+current_iteration_zfill+".json")),False,True)
training_json["start_lr"] = 0.001 if "start_lr" not in globals() else start_lr
training_json["stop_lr"] = 1e-06 if "stop_lr" not in globals() else stop_lr
training_json["decay_rate"] = 0.90 if "decay_rate" not in globals() else decay_rate
training_json["decay_steps"] = 5000 if "decay_steps" not in globals() else decay_steps
training_json["numb_steps"] = 400000 if "numb_steps" not in globals() else numb_steps
training_json["numb_test"] = 0 if "numb_test" not in globals() else numb_test
training_json["use_initial_datasets"] = True if "use_initial_datasets" not in globals() else use_initial_datasets
training_json["use_extra_datasets"] = False if "use_extra_datasets" not in globals() else use_extra_datasets
training_json["deepmd_model_version"] = 2.1 if "deepmd_model_version" not in globals() else deepmd_model_version
training_json["deepmd_model_type_descriptor"] = "se_e2_a" if "deepmd_model_type_descriptor" not in globals() else deepmd_model_type_descriptor
### #35
training_json["cluster"] = cluster
training_json["project_name"] = "nvs" if "project_name" not in globals() else project_name
training_json["allocation_name"] = "v100" if "allocation_name" not in globals() else allocation_name
training_json["arch_name"] = "v100" if "arch_name" not in globals() else arch_name

project_name = training_json["project_name"]
allocation_name = training_json["allocation_name"]
arch_name = training_json["arch_name"]
if arch_name == "v100" or arch_name == "a100":
    arch_type ="gpu"

### #35
cf.check_file(jobs_apath/("job_deepmd_train_"+arch_type +"_"+cluster+".sh"),True,True,"No SLURM file present for the training step on this cluster.")
slurm_file_master = cf.read_file(jobs_apath/("job_deepmd_train_"+arch_type +"_"+cluster+".sh"))
slurm_email = "" if "slurm_email" not in globals() else slurm_email
del jobs_apath

### Check DeePMD version
if training_json["deepmd_model_version"] not in [1.1, 1.3, 2.0, 2.1]:
    logging.critical("Invalid deepmd model version (1.1, 1.3, 2.0 or 2.1): "+str(training_json["deepmd_model_version"]))
    logging.critical("Aborting...")
    sys.exit(1)
### Check DeePMD descriptor type
if training_json["deepmd_model_type_descriptor"] not in ["se_a", "se_ar", "se_e2_a"]:
    logging.critical("Invalid deepmd type descriptor (se_a (se_e2_a) or se_ar: "+str(training_json["deepmd_model_type_descriptor"]))
    logging.critical("Aborting...")
    sys.exit(1)

### Check mismatch between cluster/arch_name/arch and DeePMD
if cluster != "jz":
    logging.critical("Only on Jean Zay !")
    logging.critical("Aborting...")
    sys.exit(1)
if training_json["deepmd_model_version"] < 2.0:
    logging.critical("Only version >= 2.0 on Jean Zay!")
    logging.critical("Aborting...")
    sys.exit(1)
if training_json["deepmd_model_version"] < 2.1 and training_json["arch_name"] == "a100":
    logging.critical("Only version >= 2.1 on Jean Zay A100 !")
    logging.critical("Aborting...")
    sys.exit(1)

### Check mismatch between DeePMD version and Descriptor
if ((training_json["deepmd_model_type_descriptor"] == "se_a") and ( training_json["deepmd_model_version"] == 1.1 ))\
or ((training_json["deepmd_model_type_descriptor"] == "se_e2_a") and ( training_json["deepmd_model_version"] == 1.1 ))\
or ((training_json["deepmd_model_type_descriptor"] == "se_ar") and ( training_json["deepmd_model_version"] == 2.0 ))\
or ((training_json["deepmd_model_type_descriptor"] == "se_ar") and ( training_json["deepmd_model_version"] == 2.1 )):
    logging.critical("Invalid DeePMD Version/Descriptor pair: "+str(training_json["deepmd_model_version"])+"/"+str(training_json["deepmd_model_type_descriptor"]))
    logging.critical("Aborting...")
    sys.exit(1)

### Descriptor name equivalence
if ((training_json["deepmd_model_type_descriptor"] == "se_a") and ( training_json["deepmd_model_version"] == 2.0 ))\
    or ((training_json["deepmd_model_type_descriptor"] == "se_a") and ( training_json["deepmd_model_version"] == 2.1 )):
        training_json["deepmd_model_type_descriptor"] = "se_e2_a"
elif ((training_json["deepmd_model_type_descriptor"] == "se_e2_a") and ( training_json["deepmd_model_version"] == 1.3 )):
        training_json["deepmd_model_type_descriptor"] = "se_a"

### Check if the default input json file exists
input_file_fpath = (training_iterative_apath/"inputs"/(str(training_json["deepmd_model_version"])+"_"+str(training_json["deepmd_model_type_descriptor"])+".json")).resolve()
training_input_json = cf.json_read(input_file_fpath,True,True)
del input_file_fpath

### Check the initial sets json file
datasets_initial_json = cf.check_initial_datasets(training_iterative_apath)

### Let us find what is in data
data_apath = training_iterative_apath/"data"
cf.check_dir(data_apath,True)
subsys_name=[]

####MAYBETODO IMPLEMENT TEST LIST FOR VALIDATION ? If DeepMD version >= 2.0

datasets_extra=[]
datasets_validation=[]
for it_data_folders in data_apath.iterdir():
    if it_data_folders.is_dir():
    ### Escape initial/extra sets, because initial get added first and extra as last
        if it_data_folders.name not in datasets_initial_json.keys() and "extra_" != it_data_folders.name[:6]:
            ### Escape test sets
            if "test_" != it_data_folders.name[:5]:
                ### Escape if set iter is superior as iter, it is only for reprocessing old stuff.
                try:
                    if int(it_data_folders.name.rsplit("_",1)[-1]) <= current_iteration:
                        subsys_name.append(it_data_folders.name.rsplit("_",1)[0])
                except:
                    pass
            else:
                datasets_validation.append(it_data_folders.name)
        ### Get the extra sets !
        elif "extra_" == it_data_folders.name[:6]:
            datasets_extra.append(it_data_folders.name)
del it_data_folders

del datasets_validation

### Training sets list construction
datasets_training=[]
datasets_training_json=[]
### Initial
nb_initial = 0
if training_json["use_initial_datasets"]:
    for it_datasets_initial_json in datasets_initial_json.keys():
        if (data_apath/it_datasets_initial_json).is_dir():
            ####MAYBEFIX: Here we don't Path because too complex
            datasets_training.append("data/"+it_datasets_initial_json+"/")
            datasets_training_json.append(it_datasets_initial_json)
            nb_initial = nb_initial+datasets_initial_json[it_datasets_initial_json]
    del it_datasets_initial_json
del datasets_initial_json

### Non-Reactive (aka subsys_nr in the initialization first) && all the others are REACTIVE !
### Total and what is added just for this iteration
nb_added_nr = 0
nb_added_r = 0
nb_added_nr_iter = 0
nb_added_r_iter = 0

### This trick remove duplicates from list via set
subsys_name = list(set(subsys_name))
subsys_name = [i for i in subsys_name if i not in config_json["subsys_nr"]]
subsys_name = [i for i in subsys_name if i not in [zzz + "-disturbed" for zzz in config_json["subsys_nr"]]]
subsys_name = sorted(subsys_name)
config_json["subsys_r"] = subsys_name
del subsys_name

if current_iteration > 0:
    for it_iteration in np.arange(1,current_iteration+1):
        try:
            for system_it in config_json["subsys_nr"]:
                if (data_apath/(system_it+"_"+str(it_iteration).zfill(3))).is_dir():
                    ####MAYBEFIX: Here we don't Path because too complex
                    datasets_training.append("data/"+system_it+"_"+str(it_iteration).zfill(3)+"/")
                    datasets_training_json.append(system_it+"_"+str(it_iteration).zfill(3))
                    nb_added_nr = nb_added_nr+np.load(str(data_apath/(system_it+"_"+str(it_iteration).zfill(3))/"set.000"/"box.npy")).shape[0]
                    if it_iteration == current_iteration:
                        nb_added_nr_iter = nb_added_nr_iter+np.load(str(data_apath/(system_it+"_"+str(it_iteration).zfill(3))/"set.000"/"box.npy")).shape[0]
            del system_it
        except(KeyError,NameError):
            pass
        try:
            for system_it in [zzz + "-disturbed" for zzz in config_json["subsys_nr"]]:
                if (data_apath/(system_it+"_"+str(it_iteration).zfill(3))).is_dir():
                    ####MAYBEFIX: Here we don't Path because too complex
                    datasets_training.append("data/"+system_it+"_"+str(it_iteration).zfill(3)+"/")
                    datasets_training_json.append(system_it+"_"+str(it_iteration).zfill(3))
                    nb_added_nr = nb_added_nr+np.load(str(data_apath/(system_it+"_"+str(it_iteration).zfill(3))/"set.000"/"box.npy")).shape[0]
                    if it_iteration == current_iteration:
                        nb_added_nr_iter = nb_added_nr_iter+np.load(str(data_apath/(system_it+"_"+str(it_iteration).zfill(3))/"set.000"/"box.npy")).shape[0]
            del system_it
        except(KeyError,NameError):
            pass
        try:
            for system_it in config_json["subsys_r"]:
                if (data_apath/(system_it+"_"+str(it_iteration).zfill(3))).is_dir():
                    ####MAYBEFIX: Here we don't Path because too complex
                    datasets_training.append("data/"+system_it+"_"+str(it_iteration).zfill(3)+"/")
                    datasets_training_json.append(system_it+"_"+str(it_iteration).zfill(3))
                    nb_added_nr = nb_added_nr+np.load(str(data_apath/(system_it+"_"+str(it_iteration).zfill(3))/"set.000"/"box.npy")).shape[0]
                    if it_iteration == current_iteration:
                        nb_added_nr_iter = nb_added_nr_iter+np.load(str(data_apath/(system_it+"_"+str(it_iteration).zfill(3))/"set.000"/"box.npy")).shape[0]
            del system_it
        except(KeyError,NameError):
            pass
    del it_iteration

### Finally the extra sets !
nb_extra = 0
if training_json["use_extra_datasets"]:
    config_json["datasets_extra"] = datasets_extra
    del datasets_extra
    for it_datasets_extra in config_json["datasets_extra"]:
        ####MAYBEFIX: Here we don't Path because too complex
        datasets_training.append("data/"+it_datasets_extra+"/")
        datasets_training_json.append(it_datasets_extra)
        nb_extra = nb_extra+np.load(str(data_apath/it_datasets_extra/"set.000"/"box.npy")).shape[0]
    del it_datasets_extra
else:
    del datasets_extra

### Total
nb_trained = nb_initial+nb_added_nr+nb_added_r+nb_extra

### Number of tests
if ( training_json["deepmd_model_version"] < 2.0 ):
    training_input_json["training"]["numb_test"] = training_json["numb_test"]

####MAYBETODO If there is validation/test sets for 2.0, maybe enforce numb_test to not 0??
####MAYBETODO If they appeared ? Maybe in the exploration extract if discarded ones, keep 10/20 to grow a validation ?

### Because changes beteween version
if ( training_json["deepmd_model_version"] >= 2.0 ):
    training_input_json["training"]["training_data"]["systems"] = datasets_training
else:
    training_input_json["training"]["systems"] = datasets_training

training_json["training_data"] = datasets_training_json
training_json["nb_trained"] = nb_trained
training_json["nb_initial"] = nb_initial
training_json["nb_added_nr"] = nb_added_nr
training_json["nb_added_r"] = nb_added_r
training_json["nb_added_nr_iter"] = nb_added_nr_iter
training_json["nb_added_r_iter"] = nb_added_r_iter
training_json["nb_extra"] = nb_extra

del datasets_training_json
del nb_trained, nb_initial, nb_extra
del nb_added_nr, nb_added_r, nb_added_nr_iter, nb_added_r_iter

### If no override, get decay steps (= nb of trained floored to the nearest 10000 divided by 4)
if "decay_steps" not in globals():
    decay_steps = cf.get_decay_steps(training_json["nb_trained"])

training_json["decay_steps"] = int(decay_steps)
decay_steps = int(decay_steps)

### THE MAGIC IS HERE
### Priority is: GOOD LUCK
if "decay_rate" in globals() and "stop_lr" not in globals():
    if "numb_steps" not in globals():
        numb_steps = training_json["numb_steps"]
    stop_lr_new = cf.get_learning_rate(numb_steps,training_json["start_lr"],decay_rate,training_json["decay_steps"])
    if "numb_steps" not in globals():
        while stop_lr_new > training_json["stop_lr"]:
            numb_steps = numb_steps+1e5
            stop_lr_new = cf.get_learning_rate(numb_steps,training_json["start_lr"],decay_rate,training_json["decay_steps"])
    training_json["numb_steps"] = int(numb_steps)
    training_json["stop_lr"] = stop_lr_new
elif "decay_rate" in globals() and "stop_lr" in globals() and "numb_steps" in globals():
    stop_lr_new = cf.get_learning_rate(numb_steps,training_json["start_lr"],decay_rate,decay_steps)
    if stop_lr_new > stop_lr:
        while stop_lr_new > stop_lr:
            decay_steps = decay_steps-1000
            stop_lr_new = cf.get_learning_rate(numb_steps,training_json["start_lr"],decay_rate,decay_steps)
    else:
        while stop_lr_new < stop_lr:
            decay_steps = decay_steps+1000
            stop_lr_new = cf.get_learning_rate(numb_steps,training_json["start_lr"],decay_rate,decay_steps)
    training_json["decay_steps"] = int(decay_steps)
    decay_rate_new = cf.get_decay_rate(numb_steps,training_json["start_lr"],stop_lr,training_json["decay_steps"])
    training_json["decay_rate"] = decay_rate_new
    del decay_rate_new
else:
    if "stop_lr" not in globals():
        stop_lr = training_json["stop_lr"]
    numb_steps = training_json["numb_steps"]
    decay_rate_new = cf.get_decay_rate(numb_steps,training_json["start_lr"],stop_lr,training_json["decay_steps"])
    while decay_rate_new < training_json["decay_rate"]:
        numb_steps = numb_steps+1e5
        decay_rate_new = cf.get_decay_rate(numb_steps,training_json["start_lr"],stop_lr,training_json["decay_steps"])
    training_json["numb_steps"] = int(numb_steps)
    training_json["decay_rate"] = decay_rate_new
    del decay_rate_new

del decay_steps, stop_lr

if ( training_json["deepmd_model_version"] >= 2.0 ):
    training_input_json["training"]["numb_steps"] = training_json["numb_steps"]
else:
    training_input_json["training"]["stop_batch"] = training_json["numb_steps"]

training_input_json["learning_rate"]["decay_steps"] = training_json["decay_steps"]

if (training_json["deepmd_model_version"] >= 1.3):
    training_input_json["learning_rate"]["stop_lr"] = training_json["stop_lr"]
else:
    training_input_json["learning_rate"]["decay_rate"] = training_json["decay_rate"]

### Set frozen/compressed bool !
training_json["is_locked"] = True
training_json["is_launched"] = False
training_json["is_checked"] = False
training_json["is_frozen"] = False
training_json["is_compressed"] = False

logging.info(training_json)
logging.info(datasets_training)

### Rsync data to local data
localdata_apath = Path(".").resolve()/"data"
cf.create_dir(localdata_apath)
for it_datasets_training in datasets_training:
    subprocess.call(["rsync","-a", str(training_iterative_apath)+"/"+it_datasets_training.rsplit("/",1)[0], str(localdata_apath)])
del it_datasets_training, localdata_apath, datasets_training

### Change some inside output
training_input_json["training"]["disp_file"]="lcurve.out"
training_input_json["training"]["save_ckpt"]="model.ckpt"

### It doesn"t exists anymore :(
if training_json["deepmd_model_version"] < 2.0:
    training_input_json["training"]["load_ckpt"]="model.ckpt"

### Create the inputs/jobfiles for each NNP with random SEED inf the form of NNP_number + random(0,1000) + current_iteration.zfil(3) so between 10000 and unlimited1000999 (at iteration 999 !!)
if current_iteration > 0:
    previous_iteration = current_iteration - 1
    previous_iteration_zfill = str(previous_iteration).zfill(3)
    prevtraining_json = cf.json_read((control_apath/("training_"+previous_iteration_zfill+".json")),True,True)
    approx_time = int(np.ceil((numb_steps*(prevtraining_json["s_per_step"]+0.25*prevtraining_json["s_per_step"])/3600)))
    del previous_iteration, previous_iteration_zfill, prevtraining_json
else:
    initial_seconds_per_1000steps = 90 if "initial_seconds_per_1000steps" not in globals() else initial_seconds_per_1000steps
    approx_time = int(np.ceil((numb_steps*initial_seconds_per_1000steps/1000/3600)))
del numb_steps

if approx_time > 100:
    approx_time = 100

for it_nnp in range(1,config_json["nb_nnp"] + 1):
    local_apath = Path(".").resolve()/str(it_nnp)
    local_apath.mkdir(exist_ok=True)
    cf.check_dir(local_apath,True)

    random.seed()
    RAND = random.randrange(0,1000)
    if training_json["deepmd_model_type_descriptor"] == "se_ar":
        training_input_json["model"]["descriptor"]["a"]["seed"] = int(str(it_nnp)+str(RAND)+current_iteration_zfill)
        training_input_json["model"]["descriptor"]["r"]["seed"] = int(str(it_nnp)+str(RAND)+current_iteration_zfill)
    else:
        training_input_json["model"]["descriptor"]["seed"] = int(str(it_nnp)+str(RAND)+current_iteration_zfill)

    training_input_json["model"]["fitting_net"]["seed"] = int(str(it_nnp)+str(RAND)+current_iteration_zfill)

    training_input_json["training"]["seed"] = int(str(it_nnp)+str(RAND)+current_iteration_zfill)

    training_input_json_fpath = Path(str(it_nnp)+"/training.json").resolve()
    cf.json_dump(training_input_json,training_input_json_fpath,False)

    slurm_file = slurm_file_master
    slurm_file = cf.replace_in_list(slurm_file,"_R_PROJECT_",project_name)
    slurm_file = cf.replace_in_list(slurm_file,"_R_WALLTIME_",str(approx_time)+":00:00")
    slurm_file = cf.replace_in_list(slurm_file,"_R_DEEPMD_MODEL_VERSION_",str(training_json["deepmd_model_version"]))
    if allocation_name == "v100":
        slurm_file = cf.replace_in_list(slurm_file,"_R_ALLOC_",allocation_name)
        if approx_time <= 20:
            if arch_name == "v100":
                slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_gpu-t3")
                slurm_file = cf.replace_in_list(slurm_file,"_R_PARTITION_","gpu_p13")
                slurm_file = cf.replace_in_list(slurm_file,"#SBATCH -C _R_SUBPARTITION_","##SBATCH -C _R_SUBPARTITION_")
            else:
                slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_gpu-t3")
                slurm_file = cf.replace_in_list(slurm_file,"_R_PARTITION_","gpu_p4")
                slurm_file = cf.replace_in_list(slurm_file,"#SBATCH -C _R_SUBPARTITION_","##SBATCH -C _R_SUBPARTITION_")
        else:
            slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_gpu-t4")
            slurm_file = cf.replace_in_list(slurm_file,"_R_PARTITION_","gpu_p13")
            slurm_file = cf.replace_in_list(slurm_file,"#SBATCH -C _R_SUBPARTITION_","##SBATCH -C _R_SUBPARTITION_")
    elif allocation_name == "a100":
        slurm_file = cf.replace_in_list(slurm_file,"_R_ALLOC_",allocation_name)
        if approx_time <= 20:
            slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_gpu-t3")
        else:
            slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_gpu-t4")
        slurm_file = cf.replace_in_list(slurm_file,"_R_PARTITION_","gpu_p5")
        slurm_file = cf.replace_in_list(slurm_file,"_R_SUBPARTITION_","a100")
    else:
        logging.critical("Unknown error. Please BUG REPORT")
        logging.critical("Aborting")
        sys.exit(1)
    if slurm_email != "":
        slurm_file = cf.replace_in_list(slurm_file,"##SBATCH --mail-type","#SBATCH --mail-type")
        slurm_file = cf.replace_in_list(slurm_file,"##SBATCH --mail-user _R_EMAIL_","#SBATCH --mail-user "+slurm_email)

    cf.write_file(local_apath/("job_deepmd_train_"+arch_type+"_"+cluster+".sh"),slurm_file)
    del slurm_file, local_apath, training_input_json_fpath, RAND
del it_nnp, approx_time, training_input_json

## Dump the config/training
cf.json_dump(config_json,(control_apath/"config.json"),True)
cf.json_dump(training_json,(control_apath/("training_"+current_iteration_zfill+".json")),True)

if "initial_seconds_per_1000steps" in globals():
    del initial_seconds_per_1000steps

logging.info("Preparation of the training is a success!")

### Cleaning
del data_apath, control_apath
del config_json, training_iterative_apath
del current_iteration, current_iteration_zfill
del training_json
del cluster, arch_type
del project_name, allocation_name, arch_name
del slurm_email
del slurm_file_master
del deepmd_iterative_apath

del sys, Path, logging, cf
del subprocess, np, random
import gc; gc.collect(); del gc
exit()