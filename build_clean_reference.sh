#!/bin/bash
set -e

# ==============================================================================
# Clean Reference Builder
#
# This script builds the "doubly clean" reference images (starless + defect-free,
# High Resolution) which are shared as the ground truth across all 7 degradation 
# combinations in the pipeline.
#
# Pipeline steps:
# 1. Runs detect_pixel_defects.py to fix hot/dead pixels on the stretched images.
# 2. Runs StarNet2 on the defect-corrected images to remove stars and generate
#    the final starless references and star masks.
# ==============================================================================

CLEAN_DIR="assets/clean"
mkdir -p "$CLEAN_DIR"

echo "******** [1/2] Running detect_pixel_defects.py on the original stretched npy ********"
python detect_pixel_defects.py \
    --input-dir assets/outputs-npy \
    --mask-dir assets/outputs-mask \
    --output-dir "$CLEAN_DIR/pixelfix-npy" \
    --tiff-dir "$CLEAN_DIR/pixelfix-tiff" \
    --mask-tiff-dir "$CLEAN_DIR/defect-mask-tiff" \
    --sigma 10 \
    --save-tiff \
    --save-mask-tiff

echo "******** [2/2] Running StarNet2 on the defect-corrected TIFF ********"
mkdir -p "$CLEAN_DIR/starless-tiff"
mkdir -p "$CLEAN_DIR/starmask-tiff"

for img in "$CLEAN_DIR"/pixelfix-tiff/*.tiff; do
    basename=$(basename "$img" .tiff)
    output="$CLEAN_DIR/starless-tiff/${basename}.tif"
    mask="$CLEAN_DIR/starmask-tiff/${basename}.tif"

    if [ -f "$output" ] && [ -f "$mask" ]; then
        echo "[SKIP] $basename: already processed (starless and mask present)"
        continue
    fi

    echo "StarNet2: $basename"
    ./starnet/starnet2 \
        --input "$img" \
        --output "$output" \
        --mask "$mask"
done

echo "******** Done ********"
echo "Doubly clean reference (starless + defect-free, HR): $CLEAN_DIR/starless-tiff"
echo "To be used as --clean-dir for build_crop_manifest.py and as --target-dir for make_dataset_merged.py"
