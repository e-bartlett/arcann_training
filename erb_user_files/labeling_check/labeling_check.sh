#!/bin/bash

cd he

# Get the number of directories:
num_dirs=$(find . -maxdepth 1 -type d | wc -l)
num_dirs=$((num_dirs - 1))

echo "Number of directories: $num_dirs"

# Initialize arrays
failed_indices=()
passed_indices=()

for i in $(seq 1 $num_dirs); do
    PADDED_DIR=$(printf "%05d" "$i")

    # You can define different target files to check for
    TARGET_FILE1="2_labeling_${PADDED_DIR}-Forces.for"
    TARGET_FILE2="2_labeling_${PADDED_DIR}-Force_Eval.fe"

    if [[ -f "$PADDED_DIR/$TARGET_FILE1" && -f "$PADDED_DIR/$TARGET_FILE2" ]]; then
        echo "File(s) found in: $PADDED_DIR"
        passed_indices+=("$i")
    else
        echo "File(s) NOT found in: $PADDED_DIR"
        failed_indices+=("$i")
    fi
done

# Print and write failed indices
if [[ ${#failed_indices[@]} -gt 0 ]]; then
    echo -e "\nFound files at indices: ${passed_indices[*]}"
    echo -e "\nMissing files at indices: ${failed_indices[*]}"
    
    printf "%s\n" "${failed_indices[@]}" > failed_indices.txt
    echo "Saved failed indices to failed_indices.txt"
else
    echo -e "\nAll files found."
    > failed_indices.txt  # Create an empty file just in case
fi
