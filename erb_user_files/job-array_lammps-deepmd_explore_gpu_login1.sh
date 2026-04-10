#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Created: 2022/01/01
# Last modified: 2024/05/15
#----------------------------------------------
# You must keep the _R_VARIABLES_ in the file.
# You must keep the name file as job-array_lammps-deepmd_explore_ARCHTYPE_myHPCkeyword.sh.
#----------------------------------------------
# QoS/Partition/SubPartition
#SBATCH --partition=bigjay,thompson
#SBATCH -C _R_SUBPARTITION_
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 1
##SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
# Walltime
#SBATCH -t 100:00:00
# Merge Output/Error
#SBATCH -o LAMMPS_DeepMD.%j
#SBATCH -e LAMMPS_DeepMD.%j
# Name of job
#SBATCH -J LAMMPS_DeepMD
#SBATCH --mem=50G
# Array
#SBATCH --array=0-8%300
#

#----------------------------------------------
# This part use the job-array-params_lammps-deepmd_explore_ARCHTYPE_myHPCkeyword.lst created
# Don't forget to replace the array_line with the correct name of the file (namely ARCHTYPE and myHPCkeyword)
# The rest should not be changed
#----------------------------------------------

SLURM_ARRAY_TASK_ID_LINE=$((SLURM_ARRAY_TASK_ID + 2))
array_line=$(sed -n "${SLURM_ARRAY_TASK_ID_LINE}p" "job-array-params_lammps-deepmd_explore_gpu_login1.lst")
IFS='/' read -ra array_param <<< "${array_line}"

JOB_PATH=${array_param[0]}
JOB_PATH="${JOB_PATH%_*}/${JOB_PATH##*_}"
JOB_PATH="${JOB_PATH%_*}/${JOB_PATH##*_}"

DeepMD_MODEL_VERSION=${array_param[1]}
IFS='" "' read -r -a DeepMD_MODEL_FILES <<< "${array_param[2]}"
LAMMPS_IN_FILE=${array_param[3]}
LAMMPS_PYTHON_SCRIPT=${array_param[4]}
LAMMPS_LOG_FILE=${LAMMPS_IN_FILE/.in/.log}
LAMMPS_OUT_FILE=${LAMMPS_IN_FILE/.in/.out}
EXTRA_FILES=()
EXTRA_FILES+=("${array_param[5]}")
if [ -n "${array_param[6]}" ]; then
EXTRA_FILES+=("${array_param[6]}")
fi
if [ -n "${array_param[6]}" ]; then
IFS='" "' read -r -a PLUMED_FILES <<< "${array_param[7]}"
EXTRA_FILES+=("${PLUMED_FILES[@]}")
fi

#----------------------------------------------
# Adapt the following lines to your HPC system
# It should be the close to the job_lammps-deepmd_explore_ARCHTYPE_myHPCkeyword.sh
#----------------------------------------------

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}/${JOB_PATH}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }

# Check
[ -f "${LAMMPS_IN_FILE}" ] || { echo "${LAMMPS_IN_FILE} does not exist. Aborting..."; exit 1; }

module purge
module load conda
source /kuhpc/work/thompson/e497b540/.conda/deepmd_salted/etc/profile.d/conda.sh
conda activate /kuhpc/work/thompson/e497b540/.conda/deepmd_salted
export XLA_FLAGS=--xla_gpu_cuda_data_dir="/kuhpc/work/thompson/e497b540/.conda/deepmd_salted"

# Example if your run in a scratch folder
export TEMPWORKDIR=./JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}/${JOB_PATH}/JOB-${SLURM_JOBID}"

#WN EDIT
sed -i "s/he.lmp/${EXTRA_FILES[0]}/g" ${LAMMPS_PYTHON_SCRIPT}
#WN EDIT

#eb edit for plumed
sed -i "s/0.00054858/1000/g" ${EXTRA_FILES[0]}

cp "${LAMMPS_IN_FILE}" "${TEMPWORKDIR}" && echo "${LAMMPS_IN_FILE} copied successfully"
cp "${LAMMPS_PYTHON_SCRIPT}" "${TEMPWORKDIR}" && echo "${LAMMPS_PYTHON_SCRIPT} copied successfully"
cp "inp.yaml" "${TEMPWORKDIR}" && echo "inp.yaml copied successfully"
cp "plumed_he.dat" "${TEMPWORKDIR}" && echo "plumed_he.dat copied successfully"
for f in "${DeepMD_MODEL_FILES[@]}"; do [ -f "${f}" ] && ln -s "$(realpath "${f}")" "${TEMPWORKDIR}" && echo "${f} linked successfully"; done
for f in "${EXTRA_FILES[@]}"; do [ -f "${f}" ] && cp "${f}" "${TEMPWORKDIR}" && echo "${f} copied successfully"; done

# Go to the temporary work directory
cd "${TEMPWORKDIR}" || { echo "Could not go to ${TEMPWORKDIR}. Aborting..."; exit 1; }

echo "# [$(date)] Running LAMMPS..."
mpirun -n 8 /kuhpc/work/thompson/e497b540/.conda/deepmd_salted/bin/python "${LAMMPS_PYTHON_SCRIPT}" > "${LAMMPS_LOG_FILE}" 2>&1
echo "# [$(date)] LAMMPS finished."

# Move back data from the temporary work directory and scratch, and clean-up
if [ -f log.cite ]; then rm log.cite ; fi
find ./ -type l -delete
rm "${SLURM_SUBMIT_DIR}/data/" -rf
mv ./* "${SLURM_SUBMIT_DIR}/${JOB_PATH}"
cd "${SLURM_SUBMIT_DIR}/${JOB_PATH}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}/${JOB_PATH}. Aborting..."; exit 1; }
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

sleep 2
exit
