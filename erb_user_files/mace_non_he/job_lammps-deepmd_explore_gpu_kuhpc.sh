#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# MACE exploration job, one trajectory, for mlip_engine "mace" WITHOUT
# hydrated_electron_mode (KU HPC). Stock ArcaNN flow with DeePMD swapped for
# MACE:
#   1. lmp runs the stock-shaped <system>.in with committee model 0
#      (pair_style mliap unified _R_MACE_MODEL_ 0, Kokkos/GPU);
#   2. mace_model_devi.py (copied here by `exploration prepare`) re-evaluates
#      every printed frame with the whole committee and writes
#      model_devi_<system>_<nnp>_<iter>.out, which pair_style deepmd would
#      have written.
# Needs the mace_lammps env with `lmp` installed (README, MACE environment).
# Keep the file name job_lammps-deepmd_explore_ARCHTYPE_myHPCkeyword.sh:
# ArcaNN keys on it even though there is no DeePMD here.
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
#SBATCH -o LAMMPS_MACE.%j
#SBATCH -e LAMMPS_MACE.%j
# Name of job
#SBATCH -J LAMMPS_MACE
#SBATCH --mem=50G
#

#----------------------------------------------
# Input files (variables) - They should not be changed
#----------------------------------------------

# Committee: model 0 as mace_<nnp>_<iter>.model-mliap_lammps.pt (LAMMPS),
# the others as plain .model files.
MACE_MODEL_FILES=("_R_MODEL_FILES_")
LAMMPS_IN_FILE="_R_LAMMPS_IN_FILE_"
LAMMPS_LOG_FILE="_R_LAMMPS_LOG_FILE_"
LAMMPS_OUT_FILE="_R_LAMMPS_OUT_FILE_"

MACE_ENV="${MACE_ENV:-mace_lammps}"

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

# Run in the trajectory folder (models, data file, plumed files and
# mace_model_devi.py are already there)
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }
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
