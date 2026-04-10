#!/bin/bash

ndir=133  # Set the number of directories (change as needed)
outdir="vis" 

# Create output directory if it doesn't exist
mkdir -p "$outdir"

for ((i=0; i<ndir; i++)); do
    padded_i=$(printf "%05d" $i)  #get the padded i dir name
    echo "${padded_i}"
    
    # Copy the files to the output directory
    cp "he/${padded_i}/1_labeling_${padded_i}-SPIN_DENSITY-1_0.cube" "$outdir"  
    cp "he/${padded_i}/labeling_${padded_i}.xyz" "$outdir"
done