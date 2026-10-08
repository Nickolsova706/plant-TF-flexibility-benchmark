#!/bin/bash

# Extract positive and negative DNA sequences from labeled BED files.
#
# Usage:
#   ./08_extract_sequences.sh \
#       /path/to/TAIR10_nuclear.fas \
#       /path/to/05_labeled_datasets \
#       /path/to/06_extracted_fasta

GENOME="$1"
BASE_DIR="$2"
OUTPUT_DIR="$3"

if [[ -z "$GENOME" || -z "$BASE_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 GENOME LABELED_DATASETS OUTPUT_DIR"
    exit 1
fi

if [[ ! -f "$GENOME" ]]; then
    echo "ERROR: Genome file not found: $GENOME"
    exit 1
fi

if [[ ! -d "$BASE_DIR" ]]; then
    echo "ERROR: Labeled dataset directory not found: $BASE_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# --------------------------------------------------
# Prepare lowercase chromosome-name genome
# --------------------------------------------------

GENOME_DIR=$(dirname "$GENOME")
GENOME_BASE=$(basename "$GENOME")
LOWERCASE_GENOME="${GENOME_DIR}/${GENOME_BASE%.fas}_lc.fas"

if [[ ! -f "$LOWERCASE_GENOME" ]]; then
    echo "Creating lowercase chromosome genome:"
    echo "$LOWERCASE_GENOME"

    sed '/^>/s/Chr/chr/' "$GENOME" > "$LOWERCASE_GENOME"
else
    echo "Lowercase genome already exists:"
    echo "$LOWERCASE_GENOME"
fi

# --------------------------------------------------
# Index genome
# --------------------------------------------------

if [[ ! -f "${LOWERCASE_GENOME}.fai" ]]; then
    echo "Indexing genome..."
    samtools faidx "$LOWERCASE_GENOME"
else
    echo "Genome index already exists."
fi

echo "------------------------------------------"
echo "Starting sequence extraction..."

processed=0
missing_negative=0

total_pos_bed=0
total_pos_fasta=0
total_neg_bed=0
total_neg_fasta=0

# --------------------------------------------------
# Process each TF
# --------------------------------------------------

for pos_bed in "$BASE_DIR"/*_positives.bed; do

    [[ ! -f "$pos_bed" ]] && continue

    tf_name=$(basename "$pos_bed" "_positives.bed")

    echo "Processing TF: $tf_name"

    TF_OUT_DIR="${OUTPUT_DIR}/${tf_name}"
    mkdir -p "$TF_OUT_DIR"

    neg_bed="${BASE_DIR}/${tf_name}_negatives.bed"

    pos_fasta="${TF_OUT_DIR}/${tf_name}_positives.fasta"
    neg_fasta="${TF_OUT_DIR}/${tf_name}_negatives.fasta"

    # ----------------------------------------------
    # Positive sequences
    # ----------------------------------------------

    pos_bed_count=$(wc -l < "$pos_bed")

    bedtools getfasta \
        -fi "$LOWERCASE_GENOME" \
        -bed "$pos_bed" \
        -fo "$pos_fasta"

    pos_fasta_count=$(grep -c '^>' "$pos_fasta" 2>/dev/null || true)

    pos_skipped=$((pos_bed_count - pos_fasta_count))

    # ----------------------------------------------
    # Negative sequences
    # ----------------------------------------------

    if [[ ! -f "$neg_bed" ]]; then
        echo "  [WARNING] Negative BED file missing."
        ((missing_negative++))
        continue
    fi

    neg_bed_count=$(wc -l < "$neg_bed")

    bedtools getfasta \
        -fi "$LOWERCASE_GENOME" \
        -bed "$neg_bed" \
        -fo "$neg_fasta"

    neg_fasta_count=$(grep -c '^>' "$neg_fasta" 2>/dev/null || true)

    neg_skipped=$((neg_bed_count - neg_fasta_count))

    # ----------------------------------------------
    # Report TF-level QC
    # ----------------------------------------------

    echo "  Positive: $pos_fasta_count / $pos_bed_count extracted"

    if [[ "$pos_skipped" -gt 0 ]]; then
        echo "  Positive sequences skipped: $pos_skipped"
    fi

    echo "  Negative: $neg_fasta_count / $neg_bed_count extracted"

    if [[ "$neg_skipped" -gt 0 ]]; then
        echo "  Negative sequences skipped: $neg_skipped"
    fi

    if [[ "$pos_fasta_count" -gt 0 && "$neg_fasta_count" -gt 0 ]]; then
        echo "  [OK] Positive and negative FASTA files created."
        ((processed++))
    else
        echo "  [WARNING] One or both FASTA files are empty."
    fi

    total_pos_bed=$((total_pos_bed + pos_bed_count))
    total_pos_fasta=$((total_pos_fasta + pos_fasta_count))

    total_neg_bed=$((total_neg_bed + neg_bed_count))
    total_neg_fasta=$((total_neg_fasta + neg_fasta_count))

done

# --------------------------------------------------
# Final summary
# --------------------------------------------------

total_pos_skipped=$((total_pos_bed - total_pos_fasta))
total_neg_skipped=$((total_neg_bed - total_neg_fasta))

echo "------------------------------------------"
echo "Sequence extraction complete."

echo "Complete TFs: $processed"
echo "TFs missing negative datasets: $missing_negative"

echo ""
echo "Overall extraction summary:"
echo "Positive intervals:  $total_pos_bed"
echo "Positive sequences:  $total_pos_fasta"
echo "Positive skipped:    $total_pos_skipped"

echo "Negative intervals:  $total_neg_bed"
echo "Negative sequences:  $total_neg_fasta"
echo "Negative skipped:    $total_neg_skipped"

echo ""
echo "Output directory: $OUTPUT_DIR"
