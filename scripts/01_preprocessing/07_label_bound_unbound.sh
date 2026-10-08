#!/bin/bash

# Label FIMO motif sites as bound (positive) or unbound (negative)
# based on overlap with cleaned TF peak regions.
#
# Usage:
#   ./07_label_bound_unbound.sh \
#       /path/to/04_fimo_results \
#       /path/to/00_peak_files \
#       /path/to/05_labeled_datasets

FIMO_RESULTS="$1"
PEAK_BASE_DIR="$2"
OUTPUT_DIR="$3"

if [[ -z "$FIMO_RESULTS" || -z "$PEAK_BASE_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 FIMO_RESULTS PEAK_BASE_DIR OUTPUT_DIR"
    exit 1
fi

if [[ ! -d "$FIMO_RESULTS" ]]; then
    echo "ERROR: FIMO results directory not found: $FIMO_RESULTS"
    exit 1
fi

if [[ ! -d "$PEAK_BASE_DIR" ]]; then
    echo "ERROR: Peak directory not found: $PEAK_BASE_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Starting bound/unbound labeling..."

processed=0
skipped=0

for tf_dir in "$FIMO_RESULTS"/*; do

    [[ ! -d "$tf_dir" ]] && continue

    tf_name=$(basename "$tf_dir")
    fimo_tsv="$tf_dir/fimo.tsv"

    target_folder=$(find "$PEAK_BASE_DIR" \
        -type d -name "$tf_name" -print -quit)

    if [[ -z "$target_folder" ]]; then
        echo "[WARNING] Peak directory not found for $tf_name"
        ((skipped++))
        continue
    fi

    peak_file=$(find "$target_folder" \
        -type f -name "*clean.narrowPeak" -print -quit)

    if [[ ! -f "$fimo_tsv" || -z "$peak_file" ]]; then
        echo "[WARNING] Missing FIMO or clean peak file for $tf_name"
        ((skipped++))
        continue
    fi

    echo "Processing: $tf_name"

    temp_bed="$OUTPUT_DIR/${tf_name}_temp.bed"
    pos_bed="$OUTPUT_DIR/${tf_name}_positives.bed"
    neg_bed="$OUTPUT_DIR/${tf_name}_negatives.bed"

    # Convert FIMO hits into 100-bp windows centered on the motif.
    # Chromosome names are converted to lowercase.
    awk -v OFS="\t" '
        NR > 1 && NF >= 5 && !/^#/ && !/FIMO/ {
            mid = ($4 + $5) / 2
            s = int(mid - 50)
            e = int(mid + 50)

            if (s < 1)
                s = 1

            print tolower($3), s, e, $1, $8
        }
    ' "$fimo_tsv" > "$temp_bed"

    # Motif windows overlapping a TF peak = positive/bound.
    bedtools intersect \
        -a "$temp_bed" \
        -b "$peak_file" \
        -u > "$pos_bed"

    # Motif windows not overlapping a TF peak = negative/unbound.
    bedtools intersect \
        -a "$temp_bed" \
        -b "$peak_file" \
        -v > "$neg_bed"

    rm "$temp_bed"

    pos_count=$(wc -l < "$pos_bed")
    neg_count=$(wc -l < "$neg_bed")

    echo "  [SUCCESS] Positives: $pos_count | Negatives: $neg_count"

    ((processed++))

done

echo "------------------------------------------"
echo "Bound/unbound labeling complete."
echo "Processed TFs: $processed"
echo "Skipped TFs: $skipped"
echo "Output directory: $OUTPUT_DIR"
