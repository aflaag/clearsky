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

echo "******** Starting StarNet2 ********"
mkdir -p assets/outputs-starless
mkdir -p assets/outputs-starmask

# Il ciclo legge i TIFF a 16-bit generati da astro_stretch.py
# La mask ora è obbligatoria per permettere a build_star_library.py di funzionare
for img in assets/outputs-tiff/*.tif; do
    basename=$(basename "$img" .tif)
    output="assets/outputs-starless/${basename}.tif"
    mask="assets/outputs-starmask/${basename}.tif"
    
    if [ -f "$output" ] && [ -f "$mask" ]; then
        echo "[SKIP] $basename: già processato (starless e mask presenti)"
        continue
    fi
    
    echo "StarNet2: $basename"
    ./starnet/starnet2 \
        --input "$img" \
        --output "$output" \
        --mask "$mask"
done

echo "******** Starting build_star_library.py ********"
python build_star_library.py

echo "******** Starting inject_stars.py ********"
python inject_stars.py

echo "******** Starting make_dataset.py ********"
if [ "$SAVE_PNG" = true ]; then
    python make_dataset.py --crops-per-image 5 --save-png
else
    python make_dataset.py --crops-per-image 5
fi

echo "******** Pipeline completata ********"
