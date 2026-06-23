from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

def siril_mtf_transformation(data_channel, shadows, midtones, highlights=1.0):
    """
    Applica l'esatta equazione di trasformazione MTF presente nella documentazione di Siril:
    xp = (original - shadows) / (highlights - shadows)
    pixel = ((midtones - 1) * xp) / (((2 * midtones - 1) * xp) - midtones)
    """
    # 1. Normalizzazione lineare iniziale rispetto al punto di nero (shadows) e di bianco
    xp = (data_channel - shadows) / (highlights - shadows)
    xp = np.clip(xp, 0.0, 1.0) # Taglia i valori fuori da [0, 1]
    
    # 2. Applicazione della funzione non-lineare Midtone Transfer Function
    numerator = (midtones - 1.0) * xp
    denominator = ((2.0 * midtones - 1.0) * xp) - midtones
    
    # Gestione divisione per zero sicuro
    with np.errstate(divide='ignore', invalid='ignore'):
        stretched = np.where(denominator != 0, numerator / denominator, 0.0)
        
    return np.clip(stretched, 0.0, 1.0)

def siril_autostretch(data_array, shadowsclip=-2.8, targetbg=0.25, linked=False):
    """
    Implementa il comando ufficiale 'autostretch' di Siril 1.2.
    
    Parameters:
    - data_array: array numpy di forma (3, H, W)
    - shadowsclip: punto di clip delle ombre in unità sigma (default Siril: -2.8)
    - targetbg: valore di luminosità obiettivo dello sfondo (default Siril: 0.25)
    - linked: se True, applica lo stesso identico stretch a tutti i canali (preserva i colori reali)
    """
    num_channels, H, W = data_array.shape
    stretched_image = np.zeros_like(data_array)
    
    # Pulizia preliminare dai NaN
    clean_data = np.nan_to_num(data_array)
    
    if linked:
        # --- MODALITÀ LINKED ---
        # Calcola i parametri globali unificati (usando la media dei canali o il canale più significativo)
        median_global = np.median(clean_data)
        std_global = np.std(clean_data)
        
        shadows = median_global + (shadowsclip * std_global)
        
        # Calcola il punto medio di input normalizzato rispetto a shadows
        xp_bg = (median_global - shadows) / (1.0 - shadows)
        xp_bg = np.clip(xp_bg, 0.001, 0.999)
        
        # Formula inversa per trovare il parametro 'midtones' che mappa il background a targetbg
        midtones = (xp_bg * (1.0 - targetbg)) / (xp_bg + targetbg - (2.0 * targetbg * xp_bg))
        
        # Applica lo stesso identico stretch a tutti e 3 i canali
        for c in range(num_channels):
            stretched_image[c] = siril_mtf_transformation(clean_data[c], shadows, midtones)
            
    else:
        # --- MODALITÀ UNLINKED (Default di Siril) ---
        # Ogni canale viene calcolato e bilanciato autonomamente
        for c in range(num_channels):
            channel = clean_data[c]
            median = np.median(channel)
            std = np.std(channel)
            
            # Calcolo dello shadow point specifico del canale
            shadows = median + (shadowsclip * std)
            
            # Calcolo del midtone specifico del canale
            xp_bg = (median - shadows) / (1.0 - shadows)
            xp_bg = np.clip(xp_bg, 0.001, 0.999)
            
            midtones = (xp_bg * (1.0 - targetbg)) / (xp_bg + targetbg - (2.0 * targetbg * xp_bg))
            
            # Applica la trasformazione al canale corrente
            stretched_image[c] = siril_mtf_transformation(channel, shadows, midtones)
            
    return stretched_image

# --- SCRIPT DI ESECUZIONE ---

# 1. Carica il FITS originale dell'Hubble
hdul = fits.open("../Download/color_hst_05461_01_wfpc2_f814w_f336w_wf_sci.fits")
data = hdul[0].data

# 2. Applica l'Autostretch ufficiale di Siril (unlinked per massimizzare i dettagli di ogni filtro)
# data_stretched = siril_autostretch(data, shadowsclip=-0.2)
data_stretched = siril_autostretch(data, shadowsclip=-5, targetbg=0.5)

# 3. Trasponi in formato RGB per Matplotlib / DDPM (H, W, 3)
data_rgb = np.transpose(data_stretched, (1, 2, 0))

# 4. Configura la figura per il salvataggio
# Rimuoviamo gli assi e i margini per salvare SOLO i pixel dell'immagine
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(data_rgb, origin="lower")
ax.axis('off') # Nasconde gli assi (pixel)

# 5. SALVA L'IMMAGINE
# 'bbox_inches="tight"' e 'pad_inches=0' eliminano i bordi bianchi di Matplotlib
# 'dpi=300' garantisce un'ottima risoluzione per il tuo dataset di DDPM
plt.savefig(
    "hubble_processed_contrast1.png", 
    bbox_inches='tight', 
    pad_inches=0, 
    dpi=300
)

# Chiude la figura per liberare la memoria RAM (fondamentale se fai un loop su molte immagini)
plt.close(fig) 

print("Immagine salvata con successo!")
