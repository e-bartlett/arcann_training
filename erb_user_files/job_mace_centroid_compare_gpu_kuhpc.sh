#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Centroid model comparison (hydrated_electron_mode). Staged into
# <iter>-training/ by training/freeze.py's MACE branch as
# job_mace_centroid_compare_ARCHTYPE_myHPCkeyword.sh and sbatch'ed alongside the
# per-NNP freeze jobs. Runs compare_centroid_models.py (copied into <iter>-training/
# by `training prepare`): every NNP/centroid_<k>.model on every data/he_<j>
# labelled set, k, j <= iter -> <iter>-training/centroid_compare/. Diagnostic only;
# nothing in ArcaNN waits on it.
#----------------------------------------------
# You must keep the _R_VARIABLES_ in the file.
#----------------------------------------------
# QoS/Partition/SubPartition
#SBATCH --partition=_R_PARTITION_
#SBATCH -C _R_SUBPARTITION_
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# Walltime
#SBATCH -t _R_WALLTIME_
# Merge Output/Error
#SBATCH -o MACE_CentroidCompare.%j
#SBATCH -e MACE_CentroidCompare.%j
# Name of job
#SBATCH -J MACE_CentroidCompare
#

#----------------------------------------------
# Files / Variables - They should not be changed
#----------------------------------------------

COMPARE_SCRIPT="compare_centroid_models.py"
COMPARE_SYSTEM="_R_SYSTEM_"
COMPARE_LOG="centroid_compare.log"
MACE_ENV="${MACE_ENV:-mace_lammps}"

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

# Go where the job has been launched (<iter>-training/)
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }
[ -f "${COMPARE_SCRIPT}" ] || { echo "${COMPARE_SCRIPT} not found (training prepare copies it). Aborting..."; exit 1; }

module purge
module load compiler/gcc/11.5 cmake/3.30.3 cuda/12.8 openmpi-cuda/5.0
# Activate after module loads, and not via conda/latest's module: that puts
# the base python ahead of the env on PATH.
# shellcheck disable=SC1091
source /kuhpc/sw/conda/latest/etc/profile.d/conda.sh
conda activate "${MACE_ENV}" || { echo "Could not activate ${MACE_ENV}. Aborting..."; exit 1; }
# Keep ~/.local site-packages from shadowing the env
export PYTHONNOUSERSITE=1
echo "Node: $(hostname)  Python: $(which python3)"
nvidia-smi -L

echo "# [$(date)] ${COMPARE_SCRIPT} --system ${COMPARE_SYSTEM}"
python3 "${COMPARE_SCRIPT}" --system "${COMPARE_SYSTEM}" --device cuda > "${COMPARE_LOG}" 2>&1
STATUS=$?
grep -v -i "warn" "${COMPARE_LOG}" | tail -n 40
echo "# [$(date)] ${COMPARE_SCRIPT} finished (status ${STATUS})."
exit ${STATUS}
