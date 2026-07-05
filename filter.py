from pathlib import Path

ROOT = Path("dataset_merged")

KEEP_NAMES = {
    "color_hst_05091_44_wfpc2_f814w_f606w_wf_sci",
    "color_hst_05092_1h_wfpc2_f814w_f606w_wf_sci",
    "color_hst_05397_1h_wfpc2_f814w_f555w_f439w_wf_sci",
    "color_hst_05942_01_wfpc2_f673n_f487n_pc_sci",
}

removed = 0
kept = 0

for combo_dir in ROOT.iterdir():
    if not combo_dir.is_dir():
        continue

    # Considera solo le cartelle con "IR" nel nome
    if "IR" not in combo_dir.name:
        continue

    print(f"\nProcessing {combo_dir.name}")

    for npy_file in combo_dir.rglob("*.npy"):
        keep = any(name in npy_file.name for name in KEEP_NAMES)

        if keep:
            kept += 1
            print(f"[KEEP] {npy_file}")
        else:
            npy_file.unlink()
            removed += 1
            print(f"[DEL ] {npy_file}")

print("\n===================================")
print(f"Kept   : {kept}")
print(f"Removed: {removed}")
print("Done.")
