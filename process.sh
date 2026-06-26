#!/bin/bash
set -e

# Flag disponibili:
#   --save-png   salva PNG di debug in astro_stretch, make_masks e make_dataset
#   --starmask   salva le starmask durante il passaggio StarNet2

SAVE_PNG=false
STARMASK=false
for arg in "$@"; do
    case "$arg" in
        --save-png)  SAVE_PNG=true  ;;
        --starmask)  STARMASK=true  ;;
    esac
done

echo "******** Starting astro_stretch.py ********"
# --save-png sempre obbligatorio qui: StarNet2 richiede i PNG prodotti da questo step.
python astro_stretch.py --save-png

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
for img in assets/outputs-png/*.png; do
    basename=$(basename "$img" .png)
    output="assets/outputs-starless/${basename}.png"
    if [ -f "$output" ]; then
        echo "[SKIP] $basename: già processato"
        continue
    fi
    echo "StarNet2: $basename"
    if [ "$STARMASK" = true ]; then
        ./starnet/starnet2 \
            --input "$img" \
            --output "$output" \
            --mask "assets/outputs-starmask/${basename}.png"
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
