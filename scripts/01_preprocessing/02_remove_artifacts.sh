#!/bin/bash
# ====================================================================
# 02_remove_artifacts.sh
#
# Removes artifact regions from every per-TF narrowPeak file.
#
# SRS7 and TRP2 contain invalid negative genomic start coordinates.
# For these two TFs, negative-start records are removed before
# artifact subtraction, matching the original preprocessing workflow.
#
# Produces *_clean.narrowPeak files alongside the originals.
#
# Usage:
#   ./02_remove_artifacts.sh <peak_base_dir> <artifact_bed>
# ====================================================================

set -euo pipefail

PEAK_BASE_DIR="${1:-./00_peak_files}"
ARTIFACTS="${2:-./global_artifacts.bed}"

command -v bedtools >/dev/null 2>&1 || {
    echo "ERROR: bedtools is not installed or not in PATH."
    exit 1
}

if [[ ! -d "$PEAK_BASE_DIR" ]]; then
    echo "ERROR: Peak directory not found: $PEAK_BASE_DIR"
    exit 1
fi

if [[ ! -f "$ARTIFACTS" ]]; then
    echo "ERROR: Artifact BED file not found: $ARTIFACTS"
    exit 1
fi

echo "=============================================="
echo "Removing artifact regions from TF peak files"
echo "Peak directory : $PEAK_BASE_DIR"
echo "Artifact BED   : $ARTIFACTS"
echo "=============================================="

find "$PEAK_BASE_DIR" -type f -name "*.narrowPeak" -print0 |
while IFS= read -r -d '' peak_file; do

    # Skip files that have already been cleaned
    [[ "$peak_file" == *"_clean.narrowPeak" ]] && continue

    output_file="${peak_file%.narrowPeak}_clean.narrowPeak"

    # --------------------------------------------------------------
    # SRS7 and TRP2 correction
    # --------------------------------------------------------------
    if [[ "$peak_file" == *"/SRS7_colamp_a/"* ||
          "$peak_file" == *"/TRP2_colamp_a/"* ]]; then

        echo "Special coordinate correction: $(basename "$(dirname "$(dirname "$peak_file")")")"

        awk '$2 >= 0' "$peak_file" |
            bedtools intersect -v \
                -a stdin \
                -b "$ARTIFACTS" \
                > "$output_file"

    else

        # ----------------------------------------------------------
        # Standard artifact removal for all other TFs
        # ----------------------------------------------------------
        bedtools intersect \
            -v \
            -a "$peak_file" \
            -b "$ARTIFACTS" \
            > "$output_file"

    fi

    echo "Cleaned: $(basename "$peak_file")"

done

echo ""
echo "Artifact removal complete."
