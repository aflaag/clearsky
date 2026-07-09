#!/bin/bash
set -e

# Flag disponibili:
#   --save-png  salva PNG di debug in make_masks e make_dataset

SAVE_PNG=false
for arg in "$@"; do
    case "$arg" in
        --save-png)  SAVE_PNG=true  ;;
    esac
done

echo "******** Starting astro_stretch.py ********"
# --save-tiff è obbligatorio: genera i TIFF a 16-bit necessari a StarNet2 per evitare il banding
python astro_stretch.py --save-tiff

echo "******** Starting make_masks.py ********"
if [ "$SAVE_PNG" = true ]; then
    python make_masks.py --save-png
else
    python make_masks.py
fi

echo "******** Starting detect_pixel_defects.py ********"
python detect_pixel_defects.py --save-tiff --save-mask-tiff --sigma 10

echo "******** Starting make_dataset_sr.py ********"
if [ "$SAVE_PNG" = true ]; then
    python make_dataset_ir.py --crops-per-image 200 --save-png
else
    python make_dataset_ir.py --crops-per-image 200
fi

echo "******** Pipeline IR completata ********"
