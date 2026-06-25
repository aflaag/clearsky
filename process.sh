#!/bin/bash
set -e

echo "******** Starting astro_stretch.py ********"
python astro_stretch.py

echo "******** Starting make_masks.py ********"
python make_masks.py

echo "******** Starting StarNet2 ********"
mkdir -p assets/outputs-starless
mkdir -p assets/outputs-starmask

for img in assets/outputs-png/*.png; do
    basename=$(basename "$img" .png)
    output="assets/outputs-starless/${basename}.png"
    starmask="assets/outputs-starmask/${basename}.png"

    if [ -f "$output" ]; then
        echo "[SKIP] $basename: già processato"
        continue
    fi

    echo "StarNet2: $basename"
    # Rimossa l'opzione --quiet per rendere visibile l'avanzamento
    ./starnet/starnet2 \
        --input "$img" \
        --output "$output" \
        --mask "$starmask"
done

echo "******** Starting make_dataset.py ********"
python make_dataset.py --crops-per-image 5

echo "******** Pipeline completata ********"
