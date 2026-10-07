#!/bin/bash

# Generate a list of individual TF names from the TF family folders.
# Usage:
#   ./03_generate_tf_list.sh /path/to/00_peak_files

PEAK_DIR="$1"
OUTPUT_FILE="my_219_tfs.txt"

if [[ -z "$PEAK_DIR" ]]; then
    echo "Usage: $0 /path/to/00_peak_files"
    exit 1
fi

if [[ ! -d "$PEAK_DIR" ]]; then
    echo "ERROR: Peak directory not found: $PEAK_DIR"
    exit 1
fi

# Find individual TF directories inside each *_tnt family directory.
find "$PEAK_DIR" -mindepth 2 -maxdepth 2 -type d \
    ! -name "*_tnt" \
    -exec basename {} \; |
    sort > "$OUTPUT_FILE"

echo "TF list created: $OUTPUT_FILE"
echo "Number of TFs: $(wc -l < "$OUTPUT_FILE")"
