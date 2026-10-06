#!/bin/bash
# ====================================================================
# 02_remove_artifacts.sh
#
# Removes regions listed in the global artifact blacklist from
# every per-TF narrowPeak file.
#
# Produces *_clean.narrowPeak files alongside the originals.
#
# Usage:
#   ./02_remove_artifacts.sh <peak_base_dir> <artifact_bed>
#
# Example:
#   ./02_remove_artifacts.sh ./00_peak_files ./global_artifacts.bed
# ====================================================================

set -euo pipefail

PEAK_BASE_DIR="${1:-./00_peak_files}"
ARTIFACTS="${2:-./global_artifacts.bed}"

# --------------------------------------------------------------------
# Check required program
# --------------------------------------------------------------------

command -v bedtools >/dev/null 2>&1 || {
    echo "ERROR: bedtools is not installed or not in PATH."
    exit 1
}

# --------------------------------------------------------------------
# Check input files
# --------------------------------------------------------------------

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

# --------------------------------------------------------------------
# Clean each narrowPeak file
# --------------------------------------------------------------------

find "$PEAK_BASE_DIR" -type f -name "*.narrowPeak" -print0 |
while IFS= read -r -d '' peak_file; do

    # Skip files that have already been cleaned
    [[ "$peak_file" == *"_clean.narrowPeak" ]] && continue

    output_file="${peak_file%.narrowPeak}_clean.narrowPeak"

    bedtools intersect \
        -v \
        -a "$peak_file" \
        -b "$ARTIFACTS" \
        > "$output_file"

    echo "Cleaned: $(basename "$peak_file")"

done

echo ""
echo "Artifact removal complete."
