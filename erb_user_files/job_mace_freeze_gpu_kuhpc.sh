#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# MACE tandem FREEZE job (Chunk 6 of ARCANN_TANDEM_PLAN.md). Staged per NNP by
# training/freeze.py's MACE branch as job_mace_freeze_ARCHTYPE_myHPCkeyword.sh.
# "Freezing" = convert the trained mace_<nnp>_<iter>.model to the LAMMPS ML-IAP
# deployable with `mace_create_lammps_model --format=mliap` (cuEquivariance
# fused kernels; MD_PERFORMANCE_PLAN.md Phase 3.5), then rsync the plain .model
# and the .model-mliap_lammps.pt into NNP/. `training check_freeze` verifies the
# NNP/ copy and sets is_frozen.
#----------------------------------------------
# You must keep the _R_VARIABLES_ in the file.
#----------------------------------------------
# QoS/Partition/SubPartition
#SBATCH --partition=_R_PARTITION_
#SBATCH -C _R_SUBPARTITION_
# The machine keyword's sub-partition MUST pin  nvidia&a40 : `--format=mliap`
# runs run_e3nn_to_cueq + builds the cueq kernels on-GPU and must be the same
# Kokkos arch as inference (the ML-IAP LAMMPS build is Kokkos_ARCH_AMPERE86 /
# sm_86). A mismatched arch produces a model that fails to load in the explore
# job. If the ADA89/L40 build is ever done, pin nvidia&l40 here to match.
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
# Walltime
#SBATCH -t 0:30:00
# Merge Output/Error
#SBATCH -o MACE_Freeze.%j
#SBATCH -e MACE_Freeze.%j
# Name of job
#SBATCH -J MACE_Freeze
#SBATCH --mem=64G
#

#----------------------------------------------
# Files / Variables - They should not be changed
#----------------------------------------------

MACE_MODEL="_R_MACE_MODEL_"
MACE_NNP_DIR="_R_MACE_NNP_DIR_"
MACE_LOG="_R_MACE_LOG_"
MACE_ENV="/kuhpc/work/thompson/e497b540/.conda/mace_electron_cueq"

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

# Go where the job has been launched (the per-NNP training folder)
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }
[ -f "${MACE_MODEL}" ] || { echo "${MACE_MODEL} does not exist. Aborting..."; exit 1; }

module purge
# cuda/12.6 matches the ML-IAP Kokkos/CUDA build; gcc/openmpi harmless here.
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
    --format=csv,noheader -l 15 > "${MACE_MODEL}.freeze_gpu_poll.csv" 2>&1 &
GPU_POLLER=$!

MLIAP_MODEL="${MACE_MODEL}-mliap_lammps.pt"
echo "# [$(date)] mace_create_lammps_model --format=mliap --dtype float32 ${MACE_MODEL}"
mace_create_lammps_model --format=mliap --dtype float32 "${MACE_MODEL}" > "${MACE_LOG}" 2>&1
STATUS=$?
echo "# [$(date)] mace_create_lammps_model finished (status ${STATUS})."

kill "${GPU_POLLER}" 2>/dev/null

if [ ${STATUS} -ne 0 ] || [ ! -f "${MLIAP_MODEL}" ]; then
    echo "FAILED: no ${MLIAP_MODEL} produced (status ${STATUS})."
    exit 1
fi

rsync -a "${MACE_MODEL}" "${MLIAP_MODEL}" "${MACE_NNP_DIR}/" || { echo "FAILED: rsync to ${MACE_NNP_DIR}"; exit 1; }
echo "# [$(date)] ${MACE_MODEL} + ${MLIAP_MODEL} -> ${MACE_NNP_DIR}/"

sleep 2
exit 0
