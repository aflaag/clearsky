#!/bin/bas
set -e
SAVE_PNG=false
for arg in "$@"; do
    case "$arg" in
        --save-png)  SAVE_PNG=true  ;;
    esac
done

echo "******** Starting denoise_skimage.py (pseudo-ground-truth) ********"
python denoise_skimage.py \
    --input-dir assets/inputs \
    --output-dir assets/outputs-denoised-clean \
    --method wavelet

echo "******** Starting inject_noise.py (genera versione rumorosa sintetica) ********"
python inject_noise.py \
    --input-dir assets/inputs \
    --output-dir assets/outputs-noisy

echo "******** Starting astro_stretch.py (stretch identico su pulito e rumoroso) ********"
# input-dir = originale pulito (ancora HLA) -> diventa il TARGET
# paired-dir = versione rumorosa sintetica -> diventa l'INPUT
python astro_stretch.py --save-tiff \
    --input-dir assets/inputs \
    --paired-dir assets/outputs-noisy \
    --output-npy assets/outputs-npy-clean \
    --output-tiff assets/outputs-tiff-clean \
    --paired-output-npy assets/outputs-npy-noisy \
    --paired-output-tiff assets/outputs-tiff-noisy

echo "******** Starting make_masks.py ********"
if [ "$SAVE_PNG" = true ]; then
    python make_masks.py --save-png
else
    python make_masks.py
fi

echo "******** Starting make_dataset.py ********"
# npy-dir (input, np.load) = versione RUMOROSA
# starless-dir (target, tiff) = versione PULITA
if [ "$SAVE_PNG" = true ]; then
    python make_dataset.py \
        --npy-dir assets/outputs-npy-noisy \
        --starless-dir assets/outputs-tiff-clean \
        --mask-dir assets/outputs-mask \
        --out-dir dataset \
        --crops-per-image 5 \
        --save-png
else
    python make_dataset.py \
        --npy-dir assets/outputs-npy-noisy \
        --starless-dir assets/outputs-tiff-clean \
        --mask-dir assets/outputs-mask \
        --out-dir dataset \
        --crops-per-image 5
fi

echo "******** Pipeline completata ********"
