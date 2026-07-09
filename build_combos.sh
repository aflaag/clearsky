#!/bin/bash
set -e
# Costruisce le 7 cartelle di input per le combinazioni di degradazione,
# a partire dal riferimento doppiamente pulito (assets/clean/starless-tiff,
# generato da build_clean_reference.sh).

CLEAN_TIFF="assets/clean/starless-tiff"
MASK_DIR="assets/outputs-mask"
LIBRARY_DIR="assets/star_library"
ORIG_NPY="assets/outputs-npy"
CORRECTED_NPY="assets/clean/pixelfix-npy"
OUT="assets/combos"
SEED=42

mkdir -p "$OUT"

apply_defects () {
    python apply_defect_delta.py \
        --original-dir "$ORIG_NPY" --corrected-dir "$CORRECTED_NPY" \
        --composite-dir "$1" --out-dir "$2"
}

echo "******** [1/6] inject_stars.py (una sola volta, riusato in 4 combo) ********"
python inject_stars.py \
    --starless-dir "$CLEAN_TIFF" \
    --mask-dir "$MASK_DIR" \
    --library-dir "$LIBRARY_DIR" \
    --out-dir "$OUT/_stars_base" \
    --seed "$SEED"

echo "******** [2/6] conversione NPY->TIFF per poter degradare la base con stelle ********"
python npy_to_tiff16.py --npy-dir "$OUT/_stars_base" --out-dir "$OUT/_stars_base_tiff"

echo "******** [3/6] degrade_images.py sul riferimento pulito (senza stelle) ********"
python degrade_images.py --tiff-dir "$CLEAN_TIFF" --out-dir "$OUT/_degraded_base"

echo "******** [4/6] degrade_images.py sulla base con stelle ********"
python degrade_images.py --tiff-dir "$OUT/_stars_base_tiff" --out-dir "$OUT/_degraded_stars_base"

echo "******** [5/6] assemblaggio delle 7 combo ********"

echo "-- SR (solo stelle) --"
cp -r "$OUT/_stars_base" "$OUT/SR"

echo "-- SU (solo bassa risoluzione) --"
cp -r "$OUT/_degraded_base/npy" "$OUT/SU"

echo "-- IR (solo difetti) --"
apply_defects "$CLEAN_TIFF" "$OUT/IR"

echo "-- SR_IR (stelle + difetti) --"
apply_defects "$OUT/_stars_base" "$OUT/SR_IR"

echo "-- SR_SU (stelle + bassa risoluzione) --"
cp -r "$OUT/_degraded_stars_base/npy" "$OUT/SR_SU"

echo "-- IR_SU (difetti + bassa risoluzione) --"
apply_defects "$OUT/_degraded_base/npy" "$OUT/IR_SU"

echo "-- SR_IR_SU (tutte e tre) --"
apply_defects "$OUT/_degraded_stars_base/npy" "$OUT/SR_IR_SU"

echo "******** [6/6] Pulizia intermedi ********"
rm -rf "$OUT"/_stars_base "$OUT"/_stars_base_tiff "$OUT"/_degraded_base "$OUT"/_degraded_stars_base

echo ""
echo "******** Fatto. 7 combo pronte in $OUT ********"
ls "$OUT"
echo ""
echo "Target condiviso per make_dataset_merged.py: $CLEAN_TIFF"
echo ""
echo "A seguire:"
echo "  1. python build_crop_manifest.py --clean-dir $CLEAN_TIFF --mask-dir $MASK_DIR"
echo "  2. per ciascuna delle 7 cartelle in $OUT: python make_dataset_merged.py --input-dir $OUT/<COMBO> --target-dir $CLEAN_TIFF --out-dir dataset_merged/<COMBO>"
echo "  3. python check_combo_alignment.py dataset_merged/*"
