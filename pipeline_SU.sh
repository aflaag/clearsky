#!/bin/bash
set -e
# Flag disponibili:
#   --save-png  salva PNG di debug in degrade_images.py e make_dataset_sr.py
SAVE_PNG=false
for arg in "$@"; do
    case "$arg" in
        --save-png)  SAVE_PNG=true  ;;
    esac
done

echo "******** Starting astro_stretch.py ********"
# --save-tiff genera i TIFF HR a 16-bit. Se già eseguito per la pipeline
# di star removal, questo step riusa gli stessi file (nessun bisogno di
# rigenerarli due volte a meno che il resize/stretch non sia cambiato).
python astro_stretch.py --save-tiff

echo "******** Starting make_masks.py ********"
# Riusata la stessa maschera di validità della pipeline di star removal.
python make_masks.py

echo "******** Starting degrade_images.py ********"
if [ "$SAVE_PNG" = true ]; then
    python degrade_images.py --save-png
else
    python degrade_images.py
fi

echo "******** Starting make_dataset_su.py ********"
if [ "$SAVE_PNG" = true ]; then
    python make_dataset_su.py --crops-per-image 5 --save-png
else
    python make_dataset_su.py --crops-per-image 5
fi

echo "******** Pipeline SU completata ********"
