import numpy as np
import matplotlib.pyplot as plt

pmap = np.load("saved_map.npy")

plt.figure()
plt.imshow(pmap, cmap = "PiYG_r")
plt.colorbar()
plt.show()
