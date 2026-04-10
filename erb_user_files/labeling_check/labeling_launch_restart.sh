#!/bin/sh
#SBATCH --job-name=launch
#SBATCH --partition=bigjay,thompson,sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1gb
#SBATCH --time=6:00:00
#SBATCH --output=launch.out 

# Load the required environment
module load python/3.7

cd he

# Read failed indices from the file
if [[ ! -f ../failed_indices.txt ]]; then
    echo "Error: failed_indices.txt not found!"
    exit 1
fi

sequence=$(cat ../failed_indices.txt)

for i in $sequence
do
    PADDED_DIR=$(printf "%05d" "$i")
    
    #if you need to copy any new files to each directory, you can do so as below
    #cp ../inputs/job_labeling_XXXXX.sh $PADDED_DIR
    #cp ../inputs/2_labeling_XXXXX.inp $PADDED_DIR

    cd $PADDED_DIR
    
    #these lines update the XXXXX strings in the file names and within the files
    #mv "job_labeling_XXXXX.sh" "job_labeling_${PADDED_DIR}.sh"
    #mv "2_labeling_XXXXX.inp" "2_labeling_${PADDED_DIR}.inp"

    #sed -i "s/XXXXX/$PADDED_DIR/g" "job_labeling_${PADDED_DIR}.sh"
    #sed -i "s/XXXXX/$PADDED_DIR/g" "2_labeling_${PADDED_DIR}.inp"

    # Submit the job
    sbatch "job_labeling_${PADDED_DIR}.sh"
    cd ..
done
