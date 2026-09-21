#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# MACE tandem exploration job ARRAY (Chunk 4 of ARCANN_TANDEM_PLAN.md). One
# task per trajectory; reads job-array-params_lammps-deepmd_explore_ARCHTYPE_
# myHPCkeyword.lst. Keep the job file name -- ArcaNN keys on it -- even though
# there is no DeePMD here. Body mirrors the single-trajectory job (the
# cuEquivariance / LAMMPS ML-IAP driver he_mace_md_mliap.py).
#----------------------------------------------
# You must keep the _R_VARIABLES_ in the file.
#----------------------------------------------
# QoS/Partition/SubPartition
#SBATCH --partition=_R_PARTITION_
#SBATCH -C _R_SUBPARTITION_
# Sub-partition MUST pin nvidia&a40 (Kokkos_ARCH_AMPERE86 ML-IAP build) --
# see the single-traj job.
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
# Walltime
#SBATCH -t 6:00:00
# Merge Output/Error
#SBATCH -o LAMMPS_MACE.%j
#SBATCH -e LAMMPS_MACE.%j
# Name of job
#SBATCH -J LAMMPS_MACE
#SBATCH --mem=50G
# Array
#SBATCH --array=_R_ARRAY_START_-_R_ARRAY_END_%300
#

#----------------------------------------------
# Parse this task's line from the .lst (same PATH schema as the DeePMD job:
# PATH/version/model_files/in_file/data_file/rerun/plumed/). rerun + plumed
# are empty for MACE.
#----------------------------------------------

SLURM_ARRAY_TASK_ID_LINE=$((SLURM_ARRAY_TASK_ID + 2))
array_line=$(sed -n "${SLURM_ARRAY_TASK_ID_LINE}p" "job-array-params_lammps-deepmd_explore_gpu_login1.lst")
IFS='/' read -ra array_param <<< "${array_line}"

JOB_PATH=${array_param[0]}
JOB_PATH="${JOB_PATH%_*}/${JOB_PATH##*_}"
JOB_PATH="${JOB_PATH%_*}/${JOB_PATH##*_}"

MACE_MODEL_VERSION=${array_param[1]}
IFS='" "' read -r -a MACE_MODEL_FILES <<< "${array_param[2]}"
LAMMPS_IN_FILE=${array_param[3]}
LAMMPS_PYTHON_SCRIPT=${array_param[4]}
LAMMPS_LOG_FILE=${LAMMPS_IN_FILE/.in/.log}
DATA_FILE=${array_param[5]}

MACE_ENV="${MACE_ENV:-/kuhpc/work/thompson/e497b540/.conda/mace_electron_cueq}"
MACE_DRIVER_DIR="${MACE_DRIVER_DIR:-/kuhpc/scratch/thompson/e497b540/hydrated_electron/dataset_prep/mace_electron}"

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

cd "${SLURM_SUBMIT_DIR}/${JOB_PATH}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}/${JOB_PATH}. Aborting..."; exit 1; }
[ -f "${LAMMPS_IN_FILE}" ] || { echo "${LAMMPS_IN_FILE} does not exist. Aborting..."; exit 1; }
[ -f "${LAMMPS_PYTHON_SCRIPT}" ] || { echo "${LAMMPS_PYTHON_SCRIPT} does not exist. Aborting..."; exit 1; }

module purge
# gcc-11.5 + OpenMPI-4.1 for the `mpirun -n 1` PMI wrapper; cuda/12.6 matches
# the ML-IAP Kokkos/CUDA build (build_lammps_mliap.sh).
module load compiler/gcc/11.5 openmpi/4.1 cuda/12.6
module load conda
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MACE_ENV}" || { echo "Could not activate ${MACE_ENV}. Aborting..."; exit 1; }
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${MACE_DRIVER_DIR}:${PYTHONPATH}"

export TEMPWORKDIR=./JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}/${JOB_PATH}/JOB-${SLURM_JOBID}"

cp "${LAMMPS_IN_FILE}" "${TEMPWORKDIR}" && echo "${LAMMPS_IN_FILE} copied"
cp "${LAMMPS_PYTHON_SCRIPT}" "${TEMPWORKDIR}" && echo "${LAMMPS_PYTHON_SCRIPT} copied"
[ -f "${DATA_FILE}" ] && cp "${DATA_FILE}" "${TEMPWORKDIR}" && echo "${DATA_FILE} copied"
for f in "${MACE_MODEL_FILES[@]}"; do
    [ -e "${f}" ] && ln -s "$(realpath "${f}")" "${TEMPWORKDIR}" && echo "${f} linked"
done
for f in *.model; do
    [ -e "${f}" ] && ln -s "$(realpath "${f}")" "${TEMPWORKDIR}" && echo "${f} linked"
done

nvidia-smi -L
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used \
    --format=csv,noheader -l 30 > "${SLURM_SUBMIT_DIR}/${JOB_PATH}/gpu_poll.csv" 2>&1 &
GPU_POLLER=$!

cd "${TEMPWORKDIR}" || { echo "Could not go to ${TEMPWORKDIR}. Aborting..."; exit 1; }

echo "# [$(date)] Running MACE tandem MD..."
mpirun -n 1 "${CONDA_PREFIX}/bin/python3" "${LAMMPS_PYTHON_SCRIPT}" > "${LAMMPS_LOG_FILE}" 2>&1
echo "# [$(date)] MD finished."

kill "${GPU_POLLER}" 2>/dev/null

if [ -f log.cite ]; then rm log.cite; fi
find ./ -type l -delete
mv ./* "${SLURM_SUBMIT_DIR}/${JOB_PATH}"
cd "${SLURM_SUBMIT_DIR}/${JOB_PATH}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}/${JOB_PATH}. Aborting..."; exit 1; }
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

sleep 2
exit
