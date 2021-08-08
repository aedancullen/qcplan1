import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml

map_img = np.array(Image.open("map_source.png").transpose(Image.ROTATE_270))
map_img = map_img.astype(np.uint8)

# grayscale -> binary
map_img[map_img <= 128.] = 128
map_img[map_img > 128.] = 0

map_height = map_img.shape[0]
map_width = map_img.shape[1]

# load map yaml
with open("map_source.yaml", 'r') as yaml_stream:
    try:
        map_metadata = yaml.safe_load(yaml_stream)
        map_resolution = map_metadata['resolution']
        origin = map_metadata['origin']
    except yaml.YAMLError as ex:
        print(ex)

# calculate map parameters
orig_x = origin[0]
orig_y = origin[1]
orig_s = np.sin(origin[2])
orig_c = np.cos(origin[2])

orig_x_px = round(np.abs(orig_x) * 1/map_resolution)
orig_y_px = round(np.abs(orig_y) * 1/map_resolution)

xdim = max(orig_x_px, map_height - orig_x_px)
ydim = max(orig_y_px, map_width - orig_y_px)

pmap = np.zeros((xdim * 2, ydim * 2), dtype=np.uint8)

start_x = xdim - orig_x_px
start_y = ydim - orig_y_px

pmap[start_x:start_x + map_img.shape[0], start_y:start_y+map_img.shape[1]] = map_img

np.save("gridmap.npy", pmap)

plt.figure()
plt.imshow(pmap, cmap = "PiYG_r")
plt.colorbar()
plt.show()
