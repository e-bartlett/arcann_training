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
# You must keep the name file as job_deepmd_freeze_ARCHTYPE_myHPCkeyword.sh.
#----------------------------------------------
# Project/Account
#SBATCH --partition=_R_PARTITION_
#SBATCH -C _R_SUBPARTITION_
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
# Walltime
#SBATCH -t 6:00:00
# Merge Output/Error
#SBATCH -o DeepMD_Freeze.%j
#SBATCH -e DeepMD_Freeze.%j
# Name of job
#SBATCH -J DeepMD_Freeze
#SBATCH --mem=150G
#

#----------------------------------------------
# Files / Variables - They should not be changed
#----------------------------------------------

DeepMD_MODEL_VERSION="_R_DEEPMD_VERSION_"
DeepMD_MODEL_FILE="_R_DEEPMD_MODEL_FILE_"
DeepMD_CKPT_FILE="_R_DEEPMD_CKPT_FILE_"
DeepMD_LOG_FILE="_R_DEEPMD_LOG_FILE_"
DeepMD_OUT_FILE="_R_DEEPMD_OUTPUT_FILE_"

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }

# Check
[ -f ${DeepMD_CKPT_FILE} ] || { echo "${DeepMD_CKPT_FILE} does not exist. Aborting..."; exit 1; }

module purge
module load conda
source /kuhpc/work/thompson/e497b540/.conda/deepmd-kit/etc/profile.d/conda.sh
conda activate /kuhpc/work/thompson/e497b540/.conda/deepmd-kit
export XLA_FLAGS=--xla_gpu_cuda_data_dir="/kuhpc/work/thompson/e497b540/.conda/deepmd-kit"

# Run the DeepMD freeze
echo "# [$(date)] Running DeepMD freeze..."
dp freeze -o ${DeepMD_MODEL_FILE} --log-path ${DeepMD_LOG_FILE} > ${DeepMD_OUT_FILE} 2>&1
echo "# [$(date)] DeepMD freeze finished."

sleep 2
exit
