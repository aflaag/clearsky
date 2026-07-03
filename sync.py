import os

TIFF_DIR = "assets/outputs-tiff"
INPUTS_DIR = "assets/inputs"
OUTPUTS_MASK_DIR = "assets/outputs-mask"
OUTPUTS_NPY_DIR = "assets/outputs-npy"

# Nomi (senza estensione) dei TIFF da mantenere
valid_names = {
    os.path.splitext(f)[0]
    for f in os.listdir(TIFF_DIR)
    if f.lower().endswith((".tif", ".tiff"))
}

folders = [
    INPUTS_DIR,
    OUTPUTS_MASK_DIR,
    OUTPUTS_NPY_DIR,
]

removed = 0

for folder in folders:
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        if not os.path.isfile(path):
            continue

        basename = os.path.splitext(filename)[0]

        if basename not in valid_names:
            os.remove(path)
            removed += 1
            print(f"Removed: {path}")

print(f"\nDone. Removed {removed} files.")
print(f"Kept {len(valid_names)} samples.")
