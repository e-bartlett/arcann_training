## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""

###################################### No change past here
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")

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

current_apath = Path(".").resolve()
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]

logging.info("Deleting DP-Freeze error files...")
cf.remove_file_glob(current_apath,"**/graph*freeze.out")
logging.info("Deleting DP-Compress error files...")
cf.remove_file_glob(current_apath,"**/graph*compress.out")
logging.info("Deleting DP-Train error files...")
cf.remove_file_glob(current_apath,"**/training.out")
logging.info("Deleting SLURM launch files...")
cf.remove_file_glob(current_apath,"**/*.sh")
logging.info("Cleaning done!")

del deepmd_iterative_apath, training_iterative_apath, current_apath, current_iteration_zfill

del sys, Path, logging, cf
import gc; gc.collect(); del gc
exit()