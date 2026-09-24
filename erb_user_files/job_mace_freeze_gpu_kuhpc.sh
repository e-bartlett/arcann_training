#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# MACE tandem FREEZE job (Chunk 6 of ARCANN_TANDEM_PLAN.md). Staged per NNP by
# training/freeze.py's MACE branch as job_mace_freeze_ARCHTYPE_myHPCkeyword.sh.
# "Freezing" = convert the stage-two mace_<nnp>_<iter>_stagetwo.model to the LAMMPS ML-IAP
# deployable with `python -m mace.cli.create_lammps_model --format=mliap`
# (float64 default dtype, as in testing_mace/run_dynamics/convert_model.sh; cuEquivariance
# fused kernels; MD_PERFORMANCE_PLAN.md Phase 3.5), then rsync it and its
# -mliap_lammps.pt into NNP/ as mace_<nnp>_<iter>.model{,-mliap_lammps.pt}. `training check_freeze` verifies the
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
MACE_NNP_MODEL="_R_MACE_NNP_MODEL_"
MACE_LOG="_R_MACE_LOG_"
MACE_ENV="mace_lammps_test"

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

# Go where the job has been launched (the per-NNP training folder)
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }
[ -f "${MACE_MODEL}" ] || { echo "${MACE_MODEL} does not exist. Aborting..."; exit 1; }

module purge
# Mirrors testing_mace/run_dynamics/convert_model.sh.
module load compiler/gcc/11.5 cmake/3.30.3 cuda/12.8 openmpi-cuda/5.0
# Activate after module loads: loading conda/latest put base python ahead of the env on PATH
# shellcheck disable=SC1091
source /kuhpc/sw/conda/latest/etc/profile.d/conda.sh
conda activate "${MACE_ENV}" || { echo "Could not activate ${MACE_ENV}. Aborting..."; exit 1; }
echo "Python is $(which python)"

# GPU poller: catch a run that silently lands on the wrong card / no GPU.
nvidia-smi -L
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used \
    --format=csv,noheader -l 15 > "${MACE_MODEL}.freeze_gpu_poll.csv" 2>&1 &
GPU_POLLER=$!

MLIAP_MODEL="${MACE_MODEL}-mliap_lammps.pt"
echo "# [$(date)] python -m mace.cli.create_lammps_model ${MACE_MODEL} --format=mliap"
python -m mace.cli.create_lammps_model "${MACE_MODEL}" --format=mliap > "${MACE_LOG}" 2>&1
STATUS=$?
echo "# [$(date)] create_lammps_model finished (status ${STATUS})."

kill "${GPU_POLLER}" 2>/dev/null

if [ ${STATUS} -ne 0 ] || [ ! -f "${MLIAP_MODEL}" ]; then
    echo "FAILED: no ${MLIAP_MODEL} produced (status ${STATUS})."
    exit 1
fi

# Deploy the stage-two model under the names the rest of ArcaNN expects.
rsync -a "${MACE_MODEL}" "${MACE_NNP_DIR}/${MACE_NNP_MODEL}" || { echo "FAILED: rsync to ${MACE_NNP_DIR}"; exit 1; }
rsync -a "${MLIAP_MODEL}" "${MACE_NNP_DIR}/${MACE_NNP_MODEL}-mliap_lammps.pt" || { echo "FAILED: rsync to ${MACE_NNP_DIR}"; exit 1; }
echo "# [$(date)] ${MACE_MODEL} (+ mliap) -> ${MACE_NNP_DIR}/${MACE_NNP_MODEL}{,-mliap_lammps.pt}"

sleep 2
exit 0
