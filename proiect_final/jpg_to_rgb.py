import cv2
import numpy as np

img = cv2.imread('resources/poza.jpg')
print(type(img[:, :, 0].shape[0]))

with open('resources/poza_rgb.shape', "bw+") as f:
    f.write(img.shape[0].to_bytes(4, 'little')), f.write(img.shape[1].to_bytes(4, 'little'))

# TODO: crop to power of 2 rows, cols
img_r = img[:, :, 0].copy().astype(dtype=np.uint8)
img_g = img[:, :, 1].copy().astype(dtype=np.uint8)
img_b = img[:, :, 2].copy().astype(dtype=np.uint8)
img_r.tofile('resources/poza_rgb.r'), img_g.tofile('resources/poza_rgb.g'), img_g.tofile('resources/poza_rgb.b')
