#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# MACE exploration job ARRAY, one task per trajectory, for mlip_engine
# "mace" WITHOUT hydrated_electron_mode (KU HPC). Same steps as the
# single-trajectory job (lmp with committee model 0, then
# mace_model_devi.py), reading the stock
# job-array-params_lammps-deepmd_explore_gpu_kuhpc.lst.
# Keep the file name: ArcaNN keys on it even though there is no DeePMD here.
#----------------------------------------------
# You must keep the _R_VARIABLES_ in the file.
#----------------------------------------------
# QoS/Partition/SubPartition
#SBATCH --partition=_R_PARTITION_
#SBATCH -C _R_SUBPARTITION_
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
# Walltime
#SBATCH -t _R_WALLTIME_
# Merge Output/Error
#SBATCH -o LAMMPS_MACE.%A_%a
#SBATCH -e LAMMPS_MACE.%A_%a
# Name of job
#SBATCH -J LAMMPS_MACE
#SBATCH --mem=50G
# Array
#SBATCH --array=_R_ARRAY_START_-_R_ARRAY_END_%300
#

#----------------------------------------------
# This task's line of the .lst (stock schema):
# PATH/version/model_files/in_file/data_file/rerun_file/plumed_files/
#----------------------------------------------

SLURM_ARRAY_TASK_ID_LINE=$((SLURM_ARRAY_TASK_ID + 2))
array_line=$(sed -n "${SLURM_ARRAY_TASK_ID_LINE}p" "job-array-params_lammps-deepmd_explore_gpu_kuhpc.lst")
IFS='/' read -ra array_param <<< "${array_line}"

JOB_PATH=${array_param[0]}
JOB_PATH="${JOB_PATH%_*}/${JOB_PATH##*_}"
JOB_PATH="${JOB_PATH%_*}/${JOB_PATH##*_}"

IFS='" "' read -r -a MACE_MODEL_FILES <<< "${array_param[2]}"
LAMMPS_IN_FILE=${array_param[3]}
LAMMPS_LOG_FILE=${LAMMPS_IN_FILE/.in/.log}
LAMMPS_OUT_FILE=${LAMMPS_IN_FILE/.in/.out}

MACE_ENV="${MACE_ENV:-mace_lammps}"

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

# Run in the trajectory folder (models, data file, plumed files and
# mace_model_devi.py are already there)
cd "${SLURM_SUBMIT_DIR}/${JOB_PATH}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}/${JOB_PATH}. Aborting..."; exit 1; }
[ -f "${LAMMPS_IN_FILE}" ] || { echo "${LAMMPS_IN_FILE} does not exist. Aborting..."; exit 1; }
[ -f mace_model_devi.py ] || { echo "mace_model_devi.py does not exist. Aborting..."; exit 1; }

# Same modules lmp was built with (CUDA runtime + MPI), before the env
module purge
module load compiler/gcc/11.5 cuda/12.8 openmpi-cuda/5.0
# shellcheck disable=SC1091
source /kuhpc/sw/conda/latest/etc/profile.d/conda.sh
conda activate "${MACE_ENV}" || { echo "Could not activate ${MACE_ENV}. Aborting..."; exit 1; }
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONHOME="${CONDA_PREFIX}"
export PYTHONNOUSERSITE=1
command -v lmp > /dev/null || { echo "No lmp in ${MACE_ENV}. Aborting..."; exit 1; }

# Names prepare.py uses (<system>_<nnp>_<iter>.in)
BASE_NAME="${LAMMPS_IN_FILE%.in}"
DEVI_DUMP="${BASE_NAME}_devi.lammpstrj"
DEVI_OUT="model_devi_${BASE_NAME}.out"
COMMITTEE=()
# Every committee member as its plain .model (empty entries come from the
# '" "' split of the model list)
for f in "${MACE_MODEL_FILES[@]}"; do [ -n "${f}" ] && COMMITTEE+=("${f%-mliap_lammps.pt}"); done

nvidia-smi -L

echo "# [$(date)] Running LAMMPS..."
mpirun -n 1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half \
    -in "${LAMMPS_IN_FILE}" -log "${LAMMPS_LOG_FILE}" -screen none > "${LAMMPS_OUT_FILE}" 2>&1
echo "# [$(date)] LAMMPS finished."

# Also after a crash: the frames up to it are still usable
echo "# [$(date)] Committee deviation (${COMMITTEE[*]})..."
if python mace_model_devi.py --lammps-in "${LAMMPS_IN_FILE}" --dump "${DEVI_DUMP}" \
    --out "${DEVI_OUT}" --models "${COMMITTEE[@]}" > "${BASE_NAME}_devi.log" 2>&1; then
    rm -f "${DEVI_DUMP}"
else
    echo "mace_model_devi.py failed, see ${BASE_NAME}_devi.log (keeping ${DEVI_DUMP})."
fi
echo "# [$(date)] Deviation finished."

if [ -f log.cite ]; then rm log.cite; fi

sleep 2
exit
