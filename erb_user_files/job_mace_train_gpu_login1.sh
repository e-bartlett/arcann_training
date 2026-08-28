#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# MACE training job (Chunk 2 of ARCANN_TANDEM_PLAN.md). Staged per NNP by
# training/prepare.py's MACE branch as job_mace_train_ARCHTYPE_myHPCkeyword.sh.
#----------------------------------------------
# You must keep the _R_VARIABLES_ in the file.
#----------------------------------------------
# QoS/Partition/SubPartition
#SBATCH --partition=_R_PARTITION_
#SBATCH -C _R_SUBPARTITION_
# The machine keyword's sub-partition MUST pin an allowed MACE GPU, i.e.
#   nvidia&(l40|a40|a100)
# (see the GPU-usage table in ARCANN_TANDEM_PLAN.md: V100 OOMs, the RTX PRO
# 6000 Blackwell is sm_120 / unsupported by this env's torch).
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
# Walltime
#SBATCH -t 6:00:00
# Merge Output/Error
#SBATCH -o MACE_Train.%j
#SBATCH -e MACE_Train.%j
# Name of job
#SBATCH -J MACE_Train
#SBATCH --mem=64G
#

#----------------------------------------------
# Files / Variables - They should not be changed
#----------------------------------------------

MACE_CONFIG="_R_MACE_CONFIG_"
MACE_NAME="_R_MACE_NAME_"
MACE_LOG="_R_MACE_LOG_"
MACE_SEED="_R_SEED_"
MACE_ENV="/kuhpc/work/thompson/e497b540/.conda/mace_electron"

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

# Go where the job has been launched (the per-NNP folder)
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }

# prepare.py's converter has already written train.xyz / valid.xyz here
[ -f "${MACE_CONFIG}" ] || { echo "${MACE_CONFIG} does not exist. Aborting..."; exit 1; }
[ -f train.xyz ] && [ -f valid.xyz ] || { echo "train.xyz / valid.xyz missing. Aborting..."; exit 1; }

module purge
module load conda
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MACE_ENV}" || { echo "Could not activate ${MACE_ENV}. Aborting..."; exit 1; }
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# GPU poller: a run that silently falls back to CPU or lands on an excluded
# card must be caught, not trusted from wall-clock (plan GPU section).
nvidia-smi -L
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used \
    --format=csv,noheader -l 30 > gpu_poll.csv 2>&1 &
GPU_POLLER=$!

echo "# [$(date)] Running mace_run_train (name=${MACE_NAME}, seed=${MACE_SEED})..."
mace_run_train \
    --config "${MACE_CONFIG}" \
    --name "${MACE_NAME}" \
    --seed "${MACE_SEED}" \
    --model_dir . \
    --log_dir . \
    --checkpoints_dir ./checkpoints \
    --results_dir ./results \
    --train_file train.xyz \
    --valid_file valid.xyz \
    > "${MACE_LOG}" 2>&1
STATUS=$?
echo "# [$(date)] mace_run_train finished (status ${STATUS})."

kill "${GPU_POLLER}" 2>/dev/null

sleep 2
exit ${STATUS}
