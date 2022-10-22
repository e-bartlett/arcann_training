###################################### No change past here
import sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")

import numpy as np

training_iterative_apath = Path("..").resolve()
### Check if the deepmd_iterative_apath is defined
deepmd_iterative_apath_error = 1
if "deepmd_iterative_apath" in globals():
    if (Path(deepmd_iterative_apath)/"scripts"/"common_functions.py").is_file():
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
sys.path.insert(0, str(Path(deepmd_iterative_apath)/"scripts"))
del deepmd_iterative_apath_error
import common_functions as cf

### Temp fix before Path/Str pass
training_iterative_apath = str(training_iterative_apath)
deepmd_iterative_apath = str(deepmd_iterative_apath)

### Read what is needed (json files)
config_json_fpath = training_iterative_apath+"/control/config.json"
config_json = cf.json_read(config_json_fpath,True,True)

current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)

training_json_fpath = training_iterative_apath+"/control/training_"+current_iteration_zfill+".json"
training_json = cf.json_read(training_json_fpath,True,True)

### Checks
if training_json["is_launched"] is False:
    logging.critical("Maybe launch the training before checking?")
    logging.critical("Aborting")
    sys.exit(1)

### Check the normal termination of the training phase
time_per_step=[]
check = 0
for it_nnp in range(1, config_json["nb_nnp"] + 1):
    cf.change_dir("./"+str(it_nnp))
    if Path("training.out").is_file():
        training_out = cf.read_file("training.out")
        if any("finished training" in s for s in training_out):
            training_out_time=[s for s in training_out if "training time" in s]
            training_out_time_split=[]
            for n in range(0,len(training_out_time)):
                training_out_time_split.append(training_out_time[n].split(" "))
                training_out_time_split[n]=" ".join(training_out_time_split[n]).split()
            if Path("model.ckpt-"+str(training_out_time_split[-1][3])+".index").is_file():
                Path("model.ckpt-"+str(training_out_time_split[-1][3])+".index").rename("model.ckpt.index")
                Path("model.ckpt-"+str(training_out_time_split[-1][3])+".meta").rename("model.ckpt.meta")
                Path("model.ckpt-"+str(training_out_time_split[-1][3])+".data-00000-of-00001").rename("model.ckpt.data-00000-of-00001")
            for n in range(0,len(training_out_time_split)):
                time_per_step.append(float(training_out_time_split[n][6]))
            del n
            step_size = float(training_out_time_split[-1][3])-float(training_out_time_split[-2][3])
            check = check + 1
        else:
            logging.critical("DP Train - ./"+str(it_nnp)+" not finished/failed")
        del training_out, training_out_time, training_out_time_split
    else:
        logging.critical("DP Train - ./"+str(it_nnp)+" still running/no outfile")
    cf.change_dir("..")
del it_nnp

if check == config_json["nb_nnp"]:
    training_json["is_checked"] = True
else:
    logging.critical("Some DP Train did not finished correctly")
    logging.critical("Please check manually before relaunching this step")
    logging.critical("Aborting...")
    sys.exit(1)
del check

if ( "time_per_step" in globals() ) and ( "step_size" in globals() ):
    training_json["avg_seconds_per_step"]=np.average(time_per_step)/(step_size)
    del time_per_step, step_size

cf.json_dump(training_json,training_json_fpath,True,"training.json")

logging.info("The training phase is a success!")

### Cleaning
del config_json, config_json_fpath, training_iterative_apath
del current_iteration, current_iteration_zfill
del training_json, training_json_fpath
del deepmd_iterative_apath

del sys, Path, logging, cf
del np
import gc; gc.collect(); del gc
exit()