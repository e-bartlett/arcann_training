#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# MACE tandem exploration job (Chunk 4 of ARCANN_TANDEM_PLAN.md). One
# trajectory: runs the MACE MD driver (<system>_<nnp>_<iter>_mace_lammps.py,
# a thin wrapper around dataset_prep/mace_electron/he_mace_md_mliap.py -- the
# cuEquivariance / LAMMPS ML-IAP path, MD_PERFORMANCE_PLAN.md Phase 3.5,
# ~0.72 ns/day on A40). Keep the job file name
# job_lammps-deepmd_explore_ARCHTYPE_myHPCkeyword.sh -- ArcaNN keys on it --
# even though there is no DeePMD here.
#----------------------------------------------
# You must keep the _R_VARIABLES_ in the file.
#----------------------------------------------
# QoS/Partition/SubPartition
#SBATCH --partition=_R_PARTITION_
#SBATCH -C _R_SUBPARTITION_
# The machine keyword's sub-partition MUST pin  nvidia&a40 : the ML-IAP
# LAMMPS build in mace_electron_cueq is Kokkos_ARCH_AMPERE86 (sm_86) and a
# mismatched GPU arch fails at model load. ADA89/L40 is deferred
# (MD_PERFORMANCE_PLAN.md Phase 3.5 step 6); if that build is done, pin
# nvidia&l40 instead and rebuild for ADA89.
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
#

#----------------------------------------------
# Input files (variables) - They should not be changed
#----------------------------------------------

# model 0 is the mace_<nnp>_<iter>.model-mliap_lammps.pt pickle (pair_style
# mliap unified); members 1..K-1 are the plain .model siblings, linked below.
MACE_MODEL_FILES=("_R_MODEL_FILES_")
LAMMPS_IN_FILE="_R_LAMMPS_IN_FILE_"
LAMMPS_PYTHON_SCRIPT="_R_LAMMPS_PYTHON_SCRIPT_"
LAMMPS_LOG_FILE="_R_LAMMPS_LOG_FILE_"
DATA_FILE="_R_DATA_FILE_"

MACE_ENV="${MACE_ENV:-/kuhpc/work/thompson/e497b540/.conda/mace_electron_cueq}"
# dir holding he_mace_md_mliap.py + mace_committee_devi.py (imported by the driver)
MACE_DRIVER_DIR="${MACE_DRIVER_DIR:-/kuhpc/scratch/thompson/e497b540/hydrated_electron/dataset_prep/mace_electron}"

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }
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
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}/JOB-${SLURM_JOBID}"

cp "${LAMMPS_IN_FILE}" "${TEMPWORKDIR}" && echo "${LAMMPS_IN_FILE} copied"
cp "${LAMMPS_PYTHON_SCRIPT}" "${TEMPWORKDIR}" && echo "${LAMMPS_PYTHON_SCRIPT} copied"
[ -f "${DATA_FILE}" ] && cp "${DATA_FILE}" "${TEMPWORKDIR}" && echo "${DATA_FILE} copied"
# model 0's .model-mliap_lammps.pt (pair_style mliap unified) ...
for f in "${MACE_MODEL_FILES[@]}"; do
    [ -e "${f}" ] && ln -s "$(realpath "${f}")" "${TEMPWORKDIR}" && echo "${f} linked"
done
# ... and every plain .model in the run dir: committee members 1..K-1 for the
# Python committee eval + centroid_<iter>.model for the electron (both
# symlinked here by exploration/{utils,prepare}.py).
for f in *.model; do
    [ -e "${f}" ] && ln -s "$(realpath "${f}")" "${TEMPWORKDIR}" && echo "${f} linked"
done

# GPU poller: a run that silently falls back to CPU or lands on an excluded
# card must be caught, not trusted from wall-clock (plan GPU section).
nvidia-smi -L
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used \
    --format=csv,noheader -l 30 > "${SLURM_SUBMIT_DIR}/gpu_poll.csv" 2>&1 &
GPU_POLLER=$!

cd "${TEMPWORKDIR}" || { echo "Could not go to ${TEMPWORKDIR}. Aborting..."; exit 1; }

echo "# [$(date)] Running MACE tandem MD..."
mpirun -n 1 "${CONDA_PREFIX}/bin/python3" "${LAMMPS_PYTHON_SCRIPT}" > "${LAMMPS_LOG_FILE}" 2>&1
echo "# [$(date)] MD finished."

kill "${GPU_POLLER}" 2>/dev/null

if [ -f log.cite ]; then rm log.cite; fi
find ./ -type l -delete
mv ./* "${SLURM_SUBMIT_DIR}"
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

sleep 2
exit
