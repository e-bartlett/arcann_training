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
# You must keep the name file as job-array_CP2K_label_ARCHTYPE_myHPCkeyword1.sh.
#----------------------------------------------
# QoS/Partition/SubPartition
#SBATCH --qos=_R_QOS_
#SBATCH --partition=bigjay,sixhour
#SBATCH -C ib
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 16
#SBATCH --cpus-per-task _R_nb_THREADSPERMPI_
#SBATCH --hint=nomultithread
# Walltime
#SBATCH -t 6:00:00
# Merge Output/Error
#SBATCH -o CP2K.%A_%a
#SBATCH -e CP2K.%A_%a
# Name of job
#SBATCH -J _R_CP2K_JOBNAME_
#SBATCH --mem=132G
# Array
#SBATCH --array=_R_ARRAY_START_-_R_ARRAY_END_%250
#SBATCH --nodelist=r16r07n01,r16r08n01,r16r09n01,r16r10n01,r16r11n01,r16r17n01,r16r18n01,r16r19n01,r16r20n01,r16r21n01,r16r22n01,r16r27n01,r16r28n01,r16r29n01,r16r30n01,r16r31n01,r17r07n01,r17r08n01,r17r09n01,r17r10n01,r17r11n01,r17r17n01,r17r18n01,r17r19n01,r17r20n01,r17r21n01,r17r22n01,r17r27n01,r17r28n01,r17r29n01,r17r30n01,r17r31n01,r18r07n01,r18r08n01,r18r09n01,r18r10n01,r18r11n01,r18r17n01,r18r18n01,r18r19n01,r18r20n01,r18r21n01,r18r22n01,r18r27n01,r18r28n01,r18r29n01,r18r30n01,r18r31n01,r19r07n01,r19r08n01,r19r09n01,r19r10n01,r19r11n01,r19r17n01,r19r18n01,r19r19n01,r19r20n01,r19r21n01,r19r22n01,r19r27n01,r19r28n01,r19r29n01,r19r30n01,r20r07n01,r20r08n01,r21r21n01,r21r22n01,r21r27n01,r21r28n01,r21r29n01,r21r30n01,r31r05n01,r31r10n01,r31r15n01,r31r20n01,r31r25n01,r31r35n01,r11r20n04,r11r22n01,r11r22n02,r11r28n02,r11r28n03,r11r28n04,r11r30n01,r11r30n02,r11r30n03,r11r30n04,r12r06n01,r12r08n01,r12r10n01,r12r12n03,r12r18n01,r12r20n01,r12r22n01,r12r26n01,r12r28n01,r12r30n01,r13r06n01,r13r08n01,r13r10n01,r13r12n01,r13r18n01,r13r20n01,r13r22n01,r13r28n01,r13r28n03,r13r28n04,r13r30n01,r13r30n02,r13r30n03,r13r30n04,r14r12n01,r14r18n01,r14r20n01,r14r20n02,r14r22n04,r14r28n02,r14r28n04,r14r30n04,r14r38n03,r14r38n04
#

#----------------------------------------------
# Input files (variables) - They should not be changed
#----------------------------------------------
SLURM_ARRAY_TASK_ID_LARGE=$((SLURM_ARRAY_TASK_ID + _R_NEW_START_))
SLURM_ARRAY_TASK_ID_PADDED=$(printf "%05d\n" "${SLURM_ARRAY_TASK_ID_LARGE}")

CP2K_IN_FILE1="1_labeling_${SLURM_ARRAY_TASK_ID_PADDED}.inp"
CP2K_OUT_FILE1="1_labeling_${SLURM_ARRAY_TASK_ID_PADDED}.out"
CP2K_IN_FILE2="2_labeling_${SLURM_ARRAY_TASK_ID_PADDED}.inp"
CP2K_OUT_FILE2="2_labeling_${SLURM_ARRAY_TASK_ID_PADDED}.out"
CP2K_XYZ_FILE="labeling_${SLURM_ARRAY_TASK_ID_PADDED}.xyz"
CP2K_WFRST_FILE="labeling_${SLURM_ARRAY_TASK_ID_PADDED}-SCF.wfn"

#----------------------------------------------
# Adapt the following lines to your HPC system
# It should be the close to the job_CP2K_label_ARCHTYPE_myHPCkeyword1.sh
# Don't forget to replace the job_labeling_array_ARCHTYPE_myHPCkeyword1.sh at the end of the file (replacling ARCHTYPE and myHPCkeyword1)
#----------------------------------------------

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID_PADDED}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID_PADDED}. Aborting..."; exit 1; }

module purge
#module load compiler/intel/24
#module load openmpi
module load cp2k/2025.1

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
