import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_single_stamp(file_path):
    print(f"--- Analisi Singolo Stamp: {Path(file_path).name} ---")
    data = np.load(file_path)
    
    # Assicuriamoci di leggere esattamente i canali corretti
    if "rgb" not in data.files or "alpha" not in data.files:
        print(f"ERRORE: Chiavi 'rgb' o 'alpha' non trovate in {data.files}")
        return

    rgb = data["rgb"]
    alpha = data["alpha"]
    
    # Normalizzazione di sicurezza [0, 1] se i dati fossero in int 0-255
    if rgb.max() > 2.0:
        rgb = rgb.astype(np.float32) / 255.0
    if alpha.max() > 2.0:
        alpha = alpha.astype(np.float32) / 255.0

    print("1. Controllo valori assoluti RGB:")
    print(f"   Min: {rgb.min():.4f}")
    print(f"   Max: {rgb.max():.4f}")
    print(f"   Median (per canale): {np.median(rgb.reshape(-1, 3), axis=0)}")

    print("\n2. Controllo purezza del fondo (dove alpha < 0.02):")
    mask = alpha < 0.02
    if np.any(mask):
        print(f"   Mean RGB nel fondo: {rgb[mask].mean(axis=0)}")
        print(f"   Max RGB nel fondo:  {rgb[mask].max():.4f}")
        
        # Se il mean o il max qui sono alti (es. > 0.05), stai iniettando "quadratini"
        if rgb[mask].max() > 0.05:
            print("   [!] ATTENZIONE: Il fondo dello stamp non è nero puro!")
    else:
        print("   [!] Nessun pixel con alpha < 0.02 trovato (lo stamp è tutto 'pieno').")

    # 3. Visualizzazione comparata
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb)
    axes[0].set_title(f"RGB\nShape: {rgb.shape}")
    axes[0].axis("off")
    
    axes[1].imshow(alpha, cmap="gray")
    axes[1].set_title("Alpha (Maschera)")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()

def analyze_library_sizes(stamps_dir):
    print(f"\n--- Analisi Dimensioni Libreria ---")
    stamp_files = list(Path(stamps_dir).glob("*.npz"))
    
    if not stamp_files:
        print(f"Nessun file .npz trovato in {stamps_dir}")
        return
        
    sizes = []
    print(f"Scansione di {len(stamp_files)} file in corso...")
    
    for p in stamp_files:
        try:
            with np.load(p) as data:
                # Carichiamo solo la shape (veloce, non decodifica l'intero array se non serve)
                h, w = data["rgb"].shape[:2]
                sizes.append(max(h, w))
        except Exception as e:
            print(f"Errore nella lettura di {p.name}: {e}")
            
    if sizes:
        sizes = np.array(sizes)
        percentiles = [10, 25, 50, 75, 90, 99]
        results = np.percentile(sizes, percentiles)
        
        print("\nDistribuzione del lato maggiore (max tra h, w):")
        for p, val in zip(percentiles, results):
            print(f"  {p}%  = {val:.0f} pixel")
        
        print(f"\n  Minimo assoluto = {sizes.min()} pixel")
        print(f"  Massimo assoluto = {sizes.max()} pixel")
        
        if results[2] > 9: # Se la mediana è maggiore di 9
            print("\n  [!] OSSERVAZIONE: La maggior parte delle stelle è più grande di 9 pixel.")
            print("  Questo spiega l'aspetto morbido e la mancanza di stelle fini (1-3 px).")

if __name__ == "__main__":
    # Aggiorna questi percorsi se necessario
    test_stamp = "./assets/star_library/stamps/star_004585.npz"
    library_dir = "./assets/star_library/stamps"
    
    if Path(test_stamp).exists():
        analyze_single_stamp(test_stamp)
    else:
        print(f"File {test_stamp} non trovato. Controlla il percorso.")
        
    if Path(library_dir).exists():
        analyze_library_sizes(library_dir)
    else:
        print(f"Directory {library_dir} non trovata.")
