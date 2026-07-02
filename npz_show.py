import numpy as np
import matplotlib.pyplot as plt

data = np.load("./assets/star_library/stamps/star_004563.npz")

print(data.files)  # mostra le chiavi

img = data[data.files[0]]  # prende il primo array

if img.max() > 1:
    img = img.astype(np.float32) / 255.0

plt.imshow(img, cmap="gray" if img.ndim == 2 else None)
plt.axis("off")
plt.show()
