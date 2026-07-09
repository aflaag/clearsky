#!/bin/bash
# Super-Resolution (SU) Pipeline Runner
#
# Executes the full sequence of scripts required to build the dataset 
# for Super-Resolution. It stretches the raw FITS, generates validity masks, 
# degrades the high-resolution images to simulate lower quality/blur, 
# and finally compiles the training patches by pairing degraded inputs 
# with HR targets.

set -e

# Available flags:
#   --save-png   saves debug PNGs in degrade_images.py and make_dataset_su.py
SAVE_PNG=false
for arg in "$@"; do
    case "$arg" in
        --save-png)  SAVE_PNG=true  ;;
    esac
done

echo "******** Starting astro_stretch.py ********"
# --save-tiff generates the 16-bit HR TIFFs. If already executed for the 
# star removal pipeline, this step reuses the same files (no need to 
# regenerate them twice unless the resize/stretch has changed).
python astro_stretch.py --save-tiff

echo "******** Starting make_masks.py ********"
# Reuses the same validity mask from the star removal pipeline.
python make_masks.py

echo "******** Starting degrade_images.py ********"
if [ "$SAVE_PNG" = true ]; then
    python degrade_images.py --save-png
else
    python degrade_images.py
fi

echo "******** Starting make_dataset_su.py ********"
if [ "$SAVE_PNG" = true ]; then
    python make_dataset_su.py --crops-per-image 200 --save-png
else
    python make_dataset_su.py --crops-per-image 200
fi

echo "******** SU Pipeline completed ********"
