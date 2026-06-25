from pathlib import Path

pngs_dir = Path("./assets/outputs-png")
target_dir = Path("./assets/inputs")

pngs_names = {f.stem for f in pngs_dir.glob("*.png")}

for target_file in target_dir.glob("*.fits"):
    if target_file.stem not in pngs_names:
        print(f"Da rimuovere: {target_file}")
        # target_file.unlink()
        # print(f"Rimosso: {target_file}")
