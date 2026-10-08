#!/bin/bash

# Run FIMO motif scanning for all selected TF motifs.
#
# Usage:
#   ./06_run_fimo.sh \
#       /path/to/02_selected_motifs \
#       /path/to/TAIR10_nuclear.fas \
#       /path/to/04_fimo_results

MOTIF_DIR="$1"
GENOME="$2"
OUTPUT_BASE="$3"

if [[ -z "$MOTIF_DIR" || -z "$GENOME" || -z "$OUTPUT_BASE" ]]; then
    echo "Usage: $0 MOTIF_DIR GENOME OUTPUT_BASE"
    exit 1
fi

if [[ ! -d "$MOTIF_DIR" ]]; then
    echo "ERROR: Motif directory not found: $MOTIF_DIR"
    exit 1
fi

if [[ ! -f "$GENOME" ]]; then
    echo "ERROR: Genome file not found: $GENOME"
    exit 1
fi

mkdir -p "$OUTPUT_BASE"

count=0
skipped=0

for tf_path in "$MOTIF_DIR"/*; do

    [[ ! -d "$tf_path" ]] && continue

    tf_name=$(basename "$tf_path")

    echo "------------------------------------------"
    echo "CURRENT TF: $tf_name"

    motif_file=$(find "$tf_path" \
        \( -name "meme_m1.txt" -o -name "dreme_d1.txt" \) \
        -print -quit)

    if [[ -z "$motif_file" ]]; then
        echo "[WARNING] No motif file found. Skipping."
        ((skipped++))
        continue
    fi

    echo "[RUN] Using motif: $(basename "$motif_file")"

    current_oc="$OUTPUT_BASE/$tf_name"
    mkdir -p "$current_oc"

    fimo \
        --thresh 5e-4 \
        --max-strand \
        --max-stored-scores 1000000 \
        --oc "$current_oc" \
        "$motif_file" \
        "$GENOME"

    if [[ -f "$current_oc/fimo.tsv" ]]; then
        echo "[OK] FIMO completed for $tf_name"
        ((count++))
    else
        echo "[WARNING] FIMO output not found for $tf_name"
    fi

done

echo "------------------------------------------"
echo "FIMO processing complete."
echo "Successful TFs: $count"
echo "Skipped TFs: $skipped"
echo "Output directory: $OUTPUT_BASE"
