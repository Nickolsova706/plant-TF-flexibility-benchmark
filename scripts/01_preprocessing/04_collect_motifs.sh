#!/bin/bash

# Collect MEME motif results for the TFs in the TF list.
#
# Usage:
#   ./04_collect_motifs.sh \
#       /path/to/motifs \
#       /path/to/TF_list.txt \
#       /path/to/02_selected_motifs

SOURCE_MOTIFS="$1"
TARGET_LIST="$2"
DEST_DIR="$3"

if [[ -z "$SOURCE_MOTIFS" || -z "$TARGET_LIST" || -z "$DEST_DIR" ]]; then
    echo "Usage: $0 SOURCE_MOTIFS TARGET_LIST DEST_DIR"
    exit 1
fi

if [[ ! -d "$SOURCE_MOTIFS" ]]; then
    echo "ERROR: Motif source directory not found: $SOURCE_MOTIFS"
    exit 1
fi

if [[ ! -f "$TARGET_LIST" ]]; then
    echo "ERROR: TF list not found: $TARGET_LIST"
    exit 1
fi

mkdir -p "$DEST_DIR"

found=0
missing=0

while read -r tf; do

    [[ -z "$tf" ]] && continue

    echo "Processing: $tf"

    motif_folder=$(find "$SOURCE_MOTIFS" -type d -name "$tf" -print -quit)

    if [[ -n "$motif_folder" && -d "$motif_folder/meme_out" ]]; then

        mkdir -p "$DEST_DIR/$tf"

        cp -r "$motif_folder/meme_out"/* "$DEST_DIR/$tf/"

        echo "  [OK] MEME results collected"
        ((found++))

    else

        echo "  [WARNING] MEME motif directory not found"
        ((missing++))

    fi

done < "$TARGET_LIST"

echo "------------------------------------------"
echo "MEME motif collection complete."
echo "TFs with MEME results: $found"
echo "TFs without MEME results: $missing"
echo "Output directory: $DEST_DIR"
