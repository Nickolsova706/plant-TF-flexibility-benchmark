#!/bin/bash
# ============================================================
# 01_build_artifact_blacklist.sh
#
# Build a global artifact blacklist from TF peak datasets.
#
# The script:
#   1. Collects all narrowPeak coordinates
#   2. Records the source peak file for each region
#   3. Sorts the coordinates
#   4. Clusters overlapping genomic regions
#   5. Counts unique source peak datasets represented
#      in each cluster
#   6. Selects clusters represented in more than the
#      specified TF threshold
#   7. Writes their genomic coordinates to
#      global_artifacts.bed
#
# Usage:
#   ./01_build_artifact_blacklist.sh <peak_dir> <output_dir> <threshold>
#
# Example:
#   ./01_build_artifact_blacklist.sh \
#       /path/to/00_peak_files \
#       /path/to/output \
#       144
# ============================================================

set -euo pipefail

PEAK_DIR="${1:-./00_peak_files}"
OUT_DIR="${2:-./}"
THRESHOLD="${3:-144}"

# ------------------------------------------------------------
# Check required programs
# ------------------------------------------------------------

command -v bedtools >/dev/null 2>&1 || {
    echo "ERROR: bedtools is not installed or not in PATH."
    exit 1
}

command -v awk >/dev/null 2>&1 || {
    echo "ERROR: awk is not available."
    exit 1
}

command -v sort >/dev/null 2>&1 || {
    echo "ERROR: sort is not available."
    exit 1
}

# ------------------------------------------------------------
# Check input directory
# ------------------------------------------------------------

if [[ ! -d "$PEAK_DIR" ]]; then
    echo "ERROR: Peak directory not found: $PEAK_DIR"
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "=============================================="
echo "Building global artifact blacklist"
echo "Peak directory  : $PEAK_DIR"
echo "Output directory: $OUT_DIR"
echo "Threshold       : > $THRESHOLD datasets"
echo "=============================================="

# ------------------------------------------------------------
# Step 1: Collect peak coordinates
#
# Column 4 stores the source peak filename.
# This follows the original preprocessing workflow.
# ------------------------------------------------------------

find "$PEAK_DIR" -type f -name "*.narrowPeak" -print0 |
while IFS= read -r -d '' peak_file; do
    awk -v source="$peak_file" '
        BEGIN { OFS="\t" }
        {
            print $1, $2, $3, source
        }
    ' "$peak_file"
done > "$OUT_DIR/all_combined.bed"

echo "[1/5] Combined peak coordinates."

# ------------------------------------------------------------
# Step 2: Sort coordinates
# ------------------------------------------------------------

sort -k1,1 -k2,2n \
    "$OUT_DIR/all_combined.bed" \
    > "$OUT_DIR/all_sorted.bed"

echo "[2/5] Sorted peak coordinates."

# ------------------------------------------------------------
# Step 3: Cluster overlapping regions
# ------------------------------------------------------------

bedtools cluster \
    -i "$OUT_DIR/all_sorted.bed" \
    > "$OUT_DIR/clustered_peaks.bed"

echo "[3/5] Overlapping peaks clustered."

# ------------------------------------------------------------
# Step 4: Identify recurrent artifact clusters
#
# A source dataset is counted only once per cluster.
# This reproduces the original:
#
#   awk '{print $5, $4}' |
#   sort | uniq |
#   cut -f1 -d' ' |
#   uniq -c |
#   awk '$1 > 144'
#
# The threshold is therefore applied to the number of
# unique source peak datasets represented in a cluster.
# ------------------------------------------------------------

awk '
BEGIN {
    FS=OFS="\t"
}
{
    print $5, $4
}
' "$OUT_DIR/clustered_peaks.bed" |
sort -k1,1 -k2,2 |
uniq |
awk '
{
    count[$1]++
}
END {
    for (cluster in count) {
        if (count[cluster] > '"$THRESHOLD"') {
            print cluster, count[cluster]
        }
    }
}
' |
sort -k1,1n \
> "$OUT_DIR/bad_clusters.txt"

n_bad=$(wc -l < "$OUT_DIR/bad_clusters.txt")

echo "[4/5] Identified $n_bad recurrent artifact clusters."

# ------------------------------------------------------------
# Step 5: Extract genomic coordinates
# ------------------------------------------------------------

awk '
BEGIN {
    FS=OFS="\t"
}
NR==FNR {
    bad[$1]=1
    next
}
($5 in bad) {
    print $1, $2, $3
}
' "$OUT_DIR/bad_clusters.txt" \
  "$OUT_DIR/clustered_peaks.bed" |
sort -k1,1 -k2,2n |
uniq \
> "$OUT_DIR/global_artifacts.bed"

n_regions=$(wc -l < "$OUT_DIR/global_artifacts.bed")

echo "[5/5] Wrote $n_regions artifact regions."

echo ""
echo "Artifact blacklist created:"
echo "$OUT_DIR/global_artifacts.bed"
echo ""
echo "Done."
