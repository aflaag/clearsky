#!/bin/bash
set -e
# Costruisce le 7 cartelle di input per le combinazioni di degradazione,
# a partire dal riferimento doppiamente pulito (assets/clean/starless-tiff,
# generato da build_clean_reference.sh).
#
# Ordine di applicazione: stelle -> bassa risoluzione -> difetti (ultimo).
# Stesso motivo pratico per cui in build_clean_reference.sh correggiamo i
# difetti PRIMA di far girare StarNet2 (rischio di scambiare un hot pixel
# per una stella): qui, al contrario, i difetti vanno reintrodotti DOPO
# aver sintetizzato le stelle, altrimenti staremmo nascondendo un difetto
# sotto una stella sintetica invece di sovrapporlo correttamente sopra.
#
# OTTIMIZZAZIONE: inject_stars.py e degrade_images.py, chiamati con lo
# stesso seed sullo stesso input, producono output IDENTICO in piu' combo
# (es. SR, SR_IR, SR_SU, SR_IR_SU condividono tutte lo stesso campo
# stellare sintetico). Invece di richiamarli 4 volte ciascuno (costoso:
# inject_stars ricarica 5000 star stamp in RAM e rifa' il posizionamento a
# griglia spaziale ad ogni run), li eseguiamo una sola volta e riusiamo
# l'output per tutte le combo che ne hanno bisogno.

CLEAN_TIFF="assets/clean/starless-tiff"
MASK_DIR="assets/outputs-mask"
LIBRARY_DIR="assets/star_library"   # default di inject_stars.py - correggi se hai usato --library-dir diverso
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
echo "Prossimi step:"
echo "  1. python build_crop_manifest.py --clean-dir $CLEAN_TIFF --mask-dir $MASK_DIR"
echo "  2. per ciascuna delle 7 cartelle in $OUT: python make_dataset_merged.py --input-dir $OUT/<COMBO> --target-dir $CLEAN_TIFF --out-dir dataset_merged/<COMBO>"
echo "  3. python check_combo_alignment.py dataset_merged/*"
