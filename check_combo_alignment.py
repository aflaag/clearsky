#!/usr/bin/env python3
"""
Verifica che tutte le cartelle combo generate da make_dataset_merged.py
contengano esattamente lo stesso insieme di stem (basename_cropindex).

Se questo controllo fallisce, il confronto "stessa patch, degradazioni
diverse" tra le 7 combinazioni non e' piu' garantito: qualche combo ha
saltato dei crop che le altre hanno mantenuto (tipicamente per mismatch
di risoluzione o file mancanti), disallineando il dataset.

Uso tipico:
    python check_combo_alignment.py dataset_merged/SR dataset_merged/IR \
        dataset_merged/SU dataset_merged/SR_IR dataset_merged/SR_SU \
        dataset_merged/IR_SU dataset_merged/SR_IR_SU
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Verifica che le cartelle combo abbiano stem identici."
    )
    parser.add_argument(
        "combo_dirs", nargs="+",
        help="Cartelle out-dir generate da make_dataset_merged.py",
    )
    args = parser.parse_args()

    stems_per_combo = {}
    for combo_dir in args.combo_dirs:
        d = Path(combo_dir) / "input" / "npy"
        stems = {p.stem for p in d.glob("*.npy")}
        stems_per_combo[combo_dir] = stems
        print(f"{combo_dir}: {len(stems)} crop")

    all_stems = set.union(*stems_per_combo.values())
    common_stems = set.intersection(*stems_per_combo.values())

    if all_stems == common_stems:
        print(f"\nOK: tutte le {len(args.combo_dirs)} combo hanno gli stessi {len(common_stems)} crop.")
        return

    print(f"\nDISALLINEAMENTO: {len(all_stems)} stem totali, solo {len(common_stems)} in comune.")
    for combo_dir, stems in stems_per_combo.items():
        missing = all_stems - stems
        if missing:
            example = sorted(missing)[:5]
            print(f"  {combo_dir}: mancano {len(missing)} crop (es. {example})")


if __name__ == "__main__":
    main()
