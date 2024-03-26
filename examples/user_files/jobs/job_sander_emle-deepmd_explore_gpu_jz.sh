#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Created: 2022/01/01
# Last modified: 2024/03/26
# Project/Account
#SBATCH --account=_R_PROJECT_@_R_ALLOC_
# QoS/Partition/SubPartition
#SBATCH --qos=_R_QOS_
#SBATCH --partition=_R_PARTITION_
#SBATCH -C _R_SUBPARTITION_
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
# Walltime
#SBATCH -t _R_WALLTIME_
# Merge Output/Error
#SBATCH -o LAMMPS_DeepMD.%j
#SBATCH -e LAMMPS_DeepMD.%j
# Name of job
#SBATCH -J LAMMPS_DeepMD
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_
#

# Input file
# Support a list of files as a bash array
DeepMD_MODEL_VERSION="_R_DEEPMD_VERSION_"
DeepMD_MODEL=("_R_MODEL_FILES_LIST_")
SANDER_IN_FILE="_R_SANDER_IN_FILE_"
EMLE_YAML_FILE="_R_EMLE_YAML_FILE_"
EXTRA_FILES=("_R_COORD_FILE_" "_R_EMLE_MODEL_FILE_" "_R_TOP_FILE_" "_R_PLUMED_FILES_LIST_")

#----------------------------------------------
# Nothing needed to be changed past this point

# Project Switch and update SCRATCH
PROJECT_NAME=${SLURM_JOB_ACCOUNT:0:3}
eval "$(idrenv -d "${PROJECT_NAME}")"
# Compare PROJECT_NAME and IDRPROJ for inequality
if [[ "${PROJECT_NAME}" != "${IDRPROJ}" ]]; then
    SCRATCH=${SCRATCH/${IDRPROJ}/${PROJECT_NAME}}
fi

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || exit 1

# Load the environment depending on the version
if [ "${SLURM_JOB_QOS:4:3}" == "gpu" ]; then
    if [ "${DeepMD_MODEL_VERSION}" == "2.1" ]; then
        module purge
        . /gpfsssd/worksf/projects/rech/kro/commun/programs/apps/deepmd-kit/2.1.5-cuda11.6_plumed-2.8.1/etc/profile.d/conda.sh
        conda activate /gpfsssd/worksf/projects/rech/kro/commun/programs/apps/deepmd-kit/2.1.5-cuda11.6_plumed-2.8.1
    else
        echo "DeePMD ${DeepMD_MODEL_VERSION} is not installed on ${SLURM_JOB_QOS}. Aborting..."; exit 1
    fi
elif [ "${SLURM_JOB_QOS:3:4}" == "cpu" ]; then
    echo "GPU on a CPU partition?? Aborting..."; exit 1
else
    echo "There is no ${SLURM_JOB_QOS}. Aborting..."; exit 1
fi

# Test if input file is present
if [ ! -f "${SANDER_IN_FILE}".in ]; then echo "No input file found. Aborting..."; exit 1; fi

# Set the temporary work directory
export TEMPWORKDIR=${SCRATCH}/JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}"/JOB-"${SLURM_JOBID}"

# Copy files to the temporary work directory
cp "${SANDER_IN_FILE}" "${TEMPWORKDIR}" && echo "${SANDER_IN_FILE} copied successfully"
cp "${SANDER_IN_FILE}" "${SANDER_IN_FILE}"."${SLURM_JOBID}"
cp "${EMLE_YAML_FILE}" "${TEMPWORKDIR}" && echo "${EMLE_YAML_FILE} copied successfully"
for f in "${EXTRA_FILES[@]}"; do [ -f "${f}" ] && cp "${f}" "${TEMPWORKDIR}" && echo "${f} copied successfully"; done
for f in "${DeepMD_MODEL[@]}"; do [ -f "${f}" ] && ln -s "$(realpath "${f}")" "${TEMPWORKDIR}" && echo "${f} linked successfully"; done
cd "${TEMPWORKDIR}" || exit 1

# MPI/OpenMP setup
echo "# [$(date)] Started"
export EXIT_CODE="0"
echo "Running on node(s): ${SLURM_NODELIST}"
echo "Running on ${SLURM_NNODES} node(s)."
# Calculate missing values
if [ -z "${SLURM_NTASKS}" ]; then
    export SLURM_NTASKS=$(( SLURM_NTASKS_PER_NODE * SLURM_NNODES))
fi
if [ -z "${SLURM_NTASKS_PER_NODE}" ]; then
    export SLURM_NTASKS_PER_NODE=$(( SLURM_NTASKS / SLURM_NNODES ))
fi
echo "Running ${SLURM_NTASKS} task(s), with ${SLURM_NTASKS_PER_NODE} task(s) per node."
echo "Running with ${SLURM_CPUS_PER_TASK} thread(s) per task."
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Launch command
emle-server --config _R_EMLE_YAML_FILE_ > server.out 2>&1 &
sander -O -i _R_SANDER_IN_FILE_ -o _R_SANDER_OUT_FILE_.out -p _R_TOP_FILE_ -c _R_COORD_FILE_ -r _R_COORD_FILE_ -x _R_SANDER_OUT_FILE_.nc

echo "# [$(date)] Ended"

# Move back data from the temporary work directory and scratch, and clean-up
if [ -f log.cite ]; then rm log.cite ; fi
find ./ -type l -delete
mv ./* "${SLURM_SUBMIT_DIR}"
cd "${SLURM_SUBMIT_DIR}" || exit 1
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }
rm "${SANDER_IN_FILE}".in."${SLURM_JOBID}"

# Done
echo "Have a nice day !"

# A small pause before SLURM savage clean-up
sleep 5
exit ${EXIT_CODE}
