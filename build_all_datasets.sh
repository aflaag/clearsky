#!/bin/bash
set -e

# ==============================================================================
# Batch Dataset Generator
#
# This script iterates through all the generated degradation combinations
# (combos) and builds the final paired datasets. For each combination, it runs 
# make_dataset_merged.py to map the degraded inputs to the shared "doubly clean" 
# starless reference targets.
# ==============================================================================

COMBOS_DIR="assets/combos"
TARGET_DIR="assets/clean/starless-tiff"
OUTPUT_ROOT="dataset_merged"

mkdir -p "$OUTPUT_ROOT"

for combo_dir in "$COMBOS_DIR"/*; do
    [ -d "$combo_dir" ] || continue

    combo_name=$(basename "$combo_dir")
    out_dir="$OUTPUT_ROOT/$combo_name"

    echo "===================================================="
    echo "Processing: $combo_name"
    echo "Input : $combo_dir"
    echo "Target: $TARGET_DIR"
    echo "Output: $out_dir"
    echo "===================================================="

    python make_dataset_merged.py \
        --input-dir "$combo_dir" \
        --target-dir "$TARGET_DIR" \
        --out-dir "$out_dir"

    echo
done

echo "All datasets have been generated."
