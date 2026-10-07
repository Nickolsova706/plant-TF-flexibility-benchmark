#!/bin/bash

# Rescue motif results for TFs that do not have MEME motifs.
#
# Usage:
#   ./05_rescue_missing_motifs.sh \
#       /path/to/motifs \
#       /path/to/03_four_rescued_dreme

SOURCE_BASE="$1"
DEST_FOUR="$2"

if [[ -z "$SOURCE_BASE" || -z "$DEST_FOUR" ]]; then
    echo "Usage: $0 SOURCE_BASE DEST_FOUR"
    exit 1
fi

if [[ ! -d "$SOURCE_BASE" ]]; then
    echo "ERROR: Motif source directory not found: $SOURCE_BASE"
    exit 1
fi

mkdir -p "$DEST_FOUR"

# Map each missing TF to the corresponding DREME family.
declare -A TF_MAP

TF_MAP["At1g68670_colamp_a"]="G2like_tnt"
TF_MAP["HSF3_colamp_a"]="HSF_tnt"
TF_MAP["LBD19_colamp_a"]="LOBAS2_tnt"
TF_MAP["RAP26_colamp_a"]="AP2EREBP_tnt"

for tf in "${!TF_MAP[@]}"; do

    family="${TF_MAP[$tf]}"
    src_dir="$SOURCE_BASE/$family/$tf"

    echo "Processing: $tf"
    echo "  Family: $family"

    if [[ ! -d "$src_dir" ]]; then
        echo "  [ERROR] Source directory not found: $src_dir"
        continue
    fi

    mkdir -p "$DEST_FOUR/$tf"

    dreme_files=$(find "$src_dir" -iname "*dreme*" -print)

    if [[ -z "$dreme_files" ]]; then
        echo "  [WARNING] No DREME files found"
        continue
    fi

    while IFS= read -r file; do
        cp -r "$file" "$DEST_FOUR/$tf/"
        echo "  [OK] Copied: $(basename "$file")"
    done <<< "$dreme_files"

done

echo "------------------------------------------"
echo "DREME rescue complete."
echo "Output directory: $DEST_FOUR"
