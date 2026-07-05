#!/bin/bash
set -e
# Costruisce il riferimento "doppiamente pulito" (starless + defect-free, HR)
# condiviso da tutte le 7 combinazioni.
#
# Ordine: prima la correzione dei difetti (sul npy stretchato originale),
# POI StarNet2 sul risultato gia' corretto - non sul TIFF originale come
# nella pipeline star-removal esistente. Motivo: se StarNet2 vede ancora
# hot pixel non corretti, rischia di scambiarli per stelle puntiformi
# (falsi positivi nella maschera stelle).
#
# La libreria di star stamps NON viene ricostruita qui: riusa quella gia'
# generata dalla pipeline star-removal esistente per inject_stars.py.

CLEAN_DIR="assets/clean"
mkdir -p "$CLEAN_DIR"

echo "******** [1/2] detect_pixel_defects.py sul npy stretchato originale ********"
python detect_pixel_defects.py \
    --input-dir assets/outputs-npy \
    --mask-dir assets/outputs-mask \
    --output-dir "$CLEAN_DIR/pixelfix-npy" \
    --tiff-dir "$CLEAN_DIR/pixelfix-tiff" \
    --mask-tiff-dir "$CLEAN_DIR/defect-mask-tiff" \
    --sigma 10 \
    --save-tiff \
    --save-mask-tiff

echo "******** [2/2] StarNet2 sul TIFF gia' corretto dai difetti ********"
mkdir -p "$CLEAN_DIR/starless-tiff"
mkdir -p "$CLEAN_DIR/starmask-tiff"

for img in "$CLEAN_DIR"/pixelfix-tiff/*.tiff; do
    basename=$(basename "$img" .tiff)
    output="$CLEAN_DIR/starless-tiff/${basename}.tif"
    mask="$CLEAN_DIR/starmask-tiff/${basename}.tif"

    if [ -f "$output" ] && [ -f "$mask" ]; then
        echo "[SKIP] $basename: gia' processato (starless e mask presenti)"
        continue
    fi

    echo "StarNet2: $basename"
    ./starnet/starnet2 \
        --input "$img" \
        --output "$output" \
        --mask "$mask"
done

echo "******** Fatto ********"
echo "Riferimento doppiamente pulito (starless + defect-free, HR): $CLEAN_DIR/starless-tiff"
echo "Usalo come --clean-dir per build_crop_manifest.py e come --target-dir per make_dataset_merged.py"
