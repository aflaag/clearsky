#!/bin/bash
set -e

# Flag disponibili:
#   --save-png  salva PNG di debug in make_masks e make_dataset
#   --starmask  salva le starmask a 16-bit durante il passaggio StarNet2

SAVE_PNG=false
STARMASK=false
for arg in "$@"; do
    case "$arg" in
        --save-png)  SAVE_PNG=true  ;;
        --starmask)  STARMASK=true  ;;
    esac
done

echo "******** Starting astro_stretch.py ********"
# --save-tiff è ora obbligatorio: genera i TIFF a 16-bit necessari a StarNet2 per evitare il banding
python astro_stretch.py --save-tiff

echo "******** Starting make_masks.py ********"
if [ "$SAVE_PNG" = true ]; then
    python make_masks.py --save-png
else
    python make_masks.py
fi

echo "******** Starting StarNet2 ********"
mkdir -p assets/outputs-starless
if [ "$STARMASK" = true ]; then
    mkdir -p assets/outputs-starmask
fi

# Il ciclo ora legge i TIFF a 16-bit generati da astro_stretch.py
for img in assets/outputs-tiff/*.tif; do
    basename=$(basename "$img" .tif)
    output="assets/outputs-starless/${basename}.tif"
    
    if [ -f "$output" ]; then
        echo "[SKIP] $basename: già processato"
        continue
    fi
    
    echo "StarNet2: $basename"
    if [ "$STARMASK" = true ]; then
        ./starnet/starnet2 \
            --input "$img" \
            --output "$output" \
            --mask "assets/outputs-starmask/${basename}.tif"
    else
        ./starnet/starnet2 \
            --input "$img" \
            --output "$output"
    fi
done

echo "******** Starting make_dataset.py ********"
if [ "$SAVE_PNG" = true ]; then
    python make_dataset.py --crops-per-image 5 --save-png
else
    python make_dataset.py --crops-per-image 5
fi

echo "******** Pipeline completata ********"
