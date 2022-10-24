## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## Project name / allocation / arch (nvs/v100/a100 or gen7156/rome/cpu)
# project_name: str = "nvs"
# allocation_name: str = "dev"
# arch_name: str = "cpu"
# slurm_email: str = ""

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
jobs_apath = deepmd_iterative_apath/"jobs"/"test"
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)
config_json = cf.json_read((control_apath/"config.json"),True,True)
test_json = cf.json_read((control_apath/("test_"+current_iteration_zfill+".json")),True,True)
current_apath = Path(".").resolve()
scripts_apath = deepmd_iterative_apath/"tools"

### Checks
if test_json["is_checked"] is False:
    logging.critical("Lock found. Run/Check first: test3_check.py")
    logging.critical("Aborting...")
    sys.exit(1)
cluster = cf.check_cluster()

### #35
test_json["cluster_2"] = cluster
test_json["project_name_2"] = project_name if "project_name" in globals() else "nvs"
test_json["allocation_name_2"] = allocation_name if "allocation_name" in globals() else "dev"
test_json["arch_name_2"] = arch_name if "arch_name" in globals() else "cpu"
project_name = test_json["project_name_2"]
allocation_name = test_json["allocation_name_2"]
arch_name = test_json["arch_name_2"]
if arch_name == "cpu":
    arch_type ="cpu"

### #35
cf.check_file(jobs_apath/("job_deepmd_test_concatenation_"+arch_type +"_"+cluster+".sh"),True,True)
slurm_file = cf.read_file(jobs_apath/("job_deepmd_test_concatenation_"+arch_type +"_"+cluster+".sh"))

slurm_file = cf.replace_in_list(slurm_file,"_R_PROJECT_",project_name)
slurm_file = cf.replace_in_list(slurm_file,"_R_WALLTIME_","02:00:00")
if allocation_name == "prepost":
    slurm_file = cf.replace_in_list(slurm_file,"_R_ALLOC_","cpu")
    slurm_file = cf.replace_in_list(slurm_file,"#SBATCH --qos=_R_QOS_","##SBATCH --qos=_R_QOS_")
    slurm_file = cf.replace_in_list(slurm_file,"_R_PARTITION_","prepost")
    slurm_file = cf.replace_in_list(slurm_file,"#SBATCH -C _R_SUBPARTITION_","##SBATCH -C _R_SUBPARTITION_")
elif allocation_name == "dev":
    slurm_file = cf.replace_in_list(slurm_file,"_R_ALLOC_","cpu")
    slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_cpu-dev")
    slurm_file = cf.replace_in_list(slurm_file,"#SBATCH --partition=_R_PARTITION_","##SBATCH --partition=_R_PARTITION_")
    slurm_file = cf.replace_in_list(slurm_file,"#SBATCH -C _R_SUBPARTITION_","##SBATCH -C _R_SUBPARTITION_")
elif allocation_name == "cpu":
    slurm_file = cf.replace_in_list(slurm_file,"_R_ALLOC_","cpu")
    slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_cpu-t3")
    slurm_file = cf.replace_in_list(slurm_file,"#SBATCH --partition=_R_PARTITION_","##SBATCH --partition=_R_PARTITION_")
    slurm_file = cf.replace_in_list(slurm_file,"#SBATCH -C _R_SUBPARTITION_","##SBATCH -C _R_SUBPARTITION_")
else:
    logging.critical("Unknown error. Please BUG REPORT")
    logging.critical("Aborting...")
    sys.exit(1)
if slurm_email != "":
    slurm_file = cf.replace_in_list(slurm_file,"##SBATCH --mail-type","#SBATCH --mail-type")
    slurm_file = cf.replace_in_list(slurm_file,"##SBATCH --mail-user _R_EMAIL_","#SBATCH --mail-user "+slurm_email)

cf.write_file(current_apath/("job_deepmd_test_concat_"+arch_type+"_"+cluster+".sh"),slurm_file)
del slurm_file

cf.check_file(scripts_apath/"_deepmd_test_concatenation.py",True,True)
python_file = cf.read_file(scripts_apath/"_deepmd_test_concatenation.py")
python_file = cf.replace_in_list(python_file,"_DEEPMD_ITERATIVE_APATH_",str(deepmd_iterative_apath))
cf.write_file(current_apath/"_deepmd_test_concatenation.py",python_file)
del python_file
logging.info("The DP-Test: concatenation-prep phase is a success!")

subprocess.call(["sbatch",str(current_apath/("job_deepmd_test_concat_"+arch_type+"_"+cluster+".sh"))])

cf.json_dump(test_json,(control_apath/("test_"+current_iteration_zfill+".json")),True)
logging.info("The DP-Test: concatenation-SLURM phase is a success!")

### Cleaning
del config_json, training_iterative_apath, control_apath, current_apath, scripts_apath, jobs_apath
del test_json
del current_iteration, current_iteration_zfill
del cluster, arch_type
del project_name, allocation_name, arch_name
del deepmd_iterative_apath
del slurm_email

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()