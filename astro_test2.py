from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

def apply_auto_stf(data_array, d=-0.75, target_mean=0.1875):
    """
    Applica lo stretch Auto-STF (stile Siril/PixInsight) canale per canale.
    
    Parameters:
    - data_array: array numpy di forma (3, H, W)
    - d: accorciamento delle ombre (shadows clipping) in unità di deviazione standard
    - target_mean: valore medio obiettivo per i mezzitoni (midtones)
    """
    stretched_channels = []
    
    for c in range(data_array.shape[0]):
        channel = np.nan_to_num(data_array[c])
        
        # 1. Calcola mediana e deviazione standard del canale
        median = np.median(channel)
        std = np.std(channel)
        
        # 2. Definisce il punto di clip delle ombre (c0) e il massimo (c1)
        c0 = median + (d * std)
        c1 = np.max(channel)
        
        # Evita divisioni per zero se il canale è piatto
        if c1 == c0:
            stretched_channels.append(np.zeros_like(channel))
            continue
            
        # 3. Normalizzazione lineare iniziale tra c0 e c1, tagliando fuori le ombre sotto c0
        normalized = (channel - c0) / (c1 - c0)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # 4. Trova il valore mediano corrente dopo il clipping
        k = np.median(normalized)
        
        # 5. Calcola il parametro dei mezzitoni (m) per mappare la mediana al target_mean
        # Equazione derivata dalla Midtone Transfer Function (MTF)
        if k + target_mean - (2 * k * target_mean) == 0:
            m = 0.5
        else:
            m = (k * (1.0 - target_mean)) / (k + target_mean - (2 * k * target_mean))
            
        # Sicurezza sui limiti di m [0, 1]
        m = np.clip(m, 0.001, 0.999)
        
        # 6. Applica la Midtone Transfer Function (MTF)
        numerator = (m - 1.0) * normalized
        denominator = ((2.0 * m - 1.0) * normalized) - m
        
        # Evita divisioni per zero durante la trasformazione non-lineare
        stretched_channel = np.divide(
            numerator, 
            denominator, 
            out=np.zeros_like(normalized), 
            where=denominator != 0
        )
        
        # Forza il range finale nell'intervallo [0, 1]
        stretched_channel = np.clip(stretched_channel, 0.0, 1.0)
        stretched_channels.append(stretched_channel)
        
    return np.array(stretched_channels)

# --- APPLICAZIONE SUI TUOI DATI ---

# 1. Carica il file FITS
# hdul = fits.open("../Download/color_hst_05461_01_wfpc2_f814w_f336w_wf_sci.fits")
hdul = fits.open("../Download/color_hlsp_legus_hst_acs_ngc5457-nw1_f814w_f555w_f435w_v1_drc_sci.fits")
data = hdul[0].data

# 2. Applica l'algoritmo con i parametri standard di Siril
# Puoi modificare d=-0.75 e target_mean=0.1875 se vuoi testare varianti
data_stretched = apply_auto_stf(data, d=-0.75, target_mean=0.1875)

# 3. Trasponi da (3, H, W) a (H, W, 3) per Matplotlib o per la tua DDPM
data_rgb = np.transpose(data_stretched, (1, 2, 0))

# 4. Mostra il risultato a schermo
plt.figure(figsize=(12, 12))
plt.imshow(data_rgb, origin="lower")
plt.axis('off')
plt.title(f"Auto-STF Stretch (Shadows: -0.75, Target Mean: 0.1875)")
plt.show()
