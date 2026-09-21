#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# MACE tandem CENTROID training job. Staged by training/prepare.py's MACE branch
# into <iter>-training/centroid/ as job_mace_centroid_ARCHTYPE_myHPCkeyword.sh
# and sbatch'ed by training/launch.py alongside the per-NNP force jobs. Trains
# CentroidMACE (user_files/train_centroid.py) float32 + cuEquivariance-native on
# <iter>-training/centroid/{train,valid}.xyz, then rsyncs centroid_<iter>.model
# into NNP/. `training check` verifies that copy and sets is_centroid_checked;
# `training check_freeze` requires NNP/centroid_<iter>.model before is_frozen.
#----------------------------------------------
# You must keep the _R_VARIABLES_ in the file.
#----------------------------------------------
# QoS/Partition/SubPartition
#SBATCH --partition=_R_PARTITION_
#SBATCH -C _R_SUBPARTITION_
# The sub-partition MUST pin nvidia&a40: the deployed centroid model runs in the
# ML-IAP explore job (Kokkos_ARCH_AMPERE86 / sm_86), and cuEquivariance kernels
# built here must match that arch. V100 also OOMs (see ARCANN_TANDEM_PLAN.md).
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
# Walltime
#SBATCH -t 6:00:00
# Merge Output/Error
#SBATCH -o MACE_Centroid.%j
#SBATCH -e MACE_Centroid.%j
# Name of job
#SBATCH -J MACE_Centroid
#SBATCH --mem=64G
#

#----------------------------------------------
# Files / Variables - They should not be changed
#----------------------------------------------

CENTROID_NAME="_R_CENTROID_NAME_"
CENTROID_INIT_FROM="_R_CENTROID_INIT_FROM_"
CENTROID_NNP_DIR="_R_CENTROID_NNP_DIR_"
CENTROID_LOG="training.log"
CENTROID_MAX_EPOCHS="${CENTROID_MAX_EPOCHS:-200}"
CENTROID_BATCH_SIZE="${CENTROID_BATCH_SIZE:-4}"
MACE_ENV="/kuhpc/work/thompson/e497b540/.conda/mace_electron_cueq"

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

# Go where the job has been launched (<iter>-training/centroid/)
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }

# train_centroid.py + model.py/loss.py/readout.py live in the work dir's user_files/
TRAINER="${SLURM_SUBMIT_DIR}/../../user_files/train_centroid.py"
[ -f "${TRAINER}" ] || { echo "${TRAINER} not found. Aborting..."; exit 1; }
[ -f train.xyz ] && [ -f valid.xyz ] || { echo "train.xyz / valid.xyz missing. Aborting..."; exit 1; }

module purge
# cuda/12.6 matches the ML-IAP Kokkos/CUDA build the deployed model runs under.
module load compiler/gcc/11.5 openmpi/4.1 cuda/12.6
module load conda
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MACE_ENV}" || { echo "Could not activate ${MACE_ENV}. Aborting..."; exit 1; }
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# GPU poller: catch a run that silently lands on the wrong card / no GPU.
nvidia-smi -L
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used \
    --format=csv,noheader -l 30 > gpu_poll.csv 2>&1 &
GPU_POLLER=$!

INIT_ARG=""
if [ -n "${CENTROID_INIT_FROM}" ]; then
    [ -f "${CENTROID_INIT_FROM}" ] || { echo "--init-from ${CENTROID_INIT_FROM} missing. Aborting..."; kill "${GPU_POLLER}" 2>/dev/null; exit 1; }
    INIT_ARG="--init-from ${CENTROID_INIT_FROM}"
fi

echo "# [$(date)] train_centroid.py --dtype float32 --cueq (name=${CENTROID_NAME}, epochs=${CENTROID_MAX_EPOCHS}) ${INIT_ARG}"
python3 "${TRAINER}" \
    --name "${CENTROID_NAME}" \
    --train-file train.xyz \
    --valid-file valid.xyz \
    --work-dir . \
    --dtype float32 \
    --cueq \
    --device cuda \
    --max-num-epochs "${CENTROID_MAX_EPOCHS}" \
    --batch-size "${CENTROID_BATCH_SIZE}" \
    --valid-batch-size "${CENTROID_BATCH_SIZE}" \
    ${INIT_ARG} \
    >> "${CENTROID_LOG}" 2>&1
STATUS=$?
echo "# [$(date)] train_centroid.py finished (status ${STATUS})."

kill "${GPU_POLLER}" 2>/dev/null

if [ ${STATUS} -ne 0 ] || [ ! -f "${CENTROID_NAME}.model" ]; then
    echo "FAILED: no ${CENTROID_NAME}.model produced (status ${STATUS})."
    exit 1
fi

rsync -a "${CENTROID_NAME}.model" "${CENTROID_NNP_DIR}/" || { echo "FAILED: rsync to ${CENTROID_NNP_DIR}"; exit 1; }
echo "# [$(date)] ${CENTROID_NAME}.model -> ${CENTROID_NNP_DIR}/"

sleep 2
exit 0
