#!/bin/bash
# Image Restoration (IR) Pipeline Runner
#
# Executes the full sequence of scripts required to build the dataset 
# for Image Restoration (e.g., repairing pixel defects).

set -e

# Available flags:
#   --save-png   saves debug PNGs in make_masks and make_dataset

SAVE_PNG=false
for arg in "$@"; do
    case "$arg" in
        --save-png)  SAVE_PNG=true  ;;
    esac
done

echo "******** Starting astro_stretch.py ********"
# --save-tiff is mandatory: it generates the 16-bit TIFFs required by StarNet2 to avoid banding
python astro_stretch.py --save-tiff

echo "******** Starting make_masks.py ********"
if [ "$SAVE_PNG" = true ]; then
    python make_masks.py --save-png
else
    python make_masks.py
fi

echo "******** Starting detect_pixel_defects.py ********"
python detect_pixel_defects.py --save-tiff --save-mask-tiff --sigma 10

echo "******** Starting make_dataset_ir.py ********"
if [ "$SAVE_PNG" = true ]; then
    python make_dataset_ir.py --crops-per-image 200 --save-png
else
    python make_dataset_ir.py --crops-per-image 200
fi

echo "******** IR Pipeline completed ********"
