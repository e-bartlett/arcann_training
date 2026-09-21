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
# You must keep the name file as job_CP2K_label_ARCHTYPE_myHPCkeyword1.sh.
#----------------------------------------------
# QoS/Partition/SubPartition
#SBATCH --qos=_R_QOS_
#SBATCH --partition=bigjay
#SBATCH -C ib,intel
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 16
#SBATCH --cpus-per-task _R_nb_THREADSPERMPI_
#SBATCH --hint=nomultithread
# Walltime
#SBATCH -t 6:00:00
# Merge Output/Error
#SBATCH -o CP2K.%j
#SBATCH -e CP2K.%j
#SBATCH --mem=132G
# Name of job
#SBATCH -J _R_CP2K_JOBNAME_
#

#----------------------------------------------
# Input files (variables) - They should not be changed
#----------------------------------------------

CP2K_IN_FILE1="1_labeling__R_PADDEDSTEP_.inp"
CP2K_OUT_FILE1="1_labeling__R_PADDEDSTEP_.out"
CP2K_IN_FILE2="2_labeling__R_PADDEDSTEP_.inp"
CP2K_OUT_FILE2="2_labeling__R_PADDEDSTEP_.out"
CP2K_XYZ_FILE="labeling__R_PADDEDSTEP_.xyz"
CP2K_WFRST_FILE="labeling__R_PADDEDSTEP_-SCF.wfn"

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

module purge
#module load compiler/intel/25
#module load intel-mpi/25
module load cp2k/2025.1
#module load ucx

echo "# [$(date)] Running CP2K first job..."
mpirun -n "${SLURM_NTASKS}" cp2k.psmp -i "${CP2K_IN_FILE1}" > "${CP2K_OUT_FILE1}"
cp "${CP2K_WFRST_FILE}" "1_${CP2K_WFRST_FILE}"
echo "# [$(date)] CP2K first job finished."
echo "# [$(date)] Running CP2K second job..."
mpirun -n "${SLURM_NTASKS}" cp2k.psmp -i "${CP2K_IN_FILE2}" > "${CP2K_OUT_FILE2}"
cp "${CP2K_WFRST_FILE}" "2_${CP2K_WFRST_FILE}"
echo "# [$(date)] CP2K second job finished."

sleep 2
exit
