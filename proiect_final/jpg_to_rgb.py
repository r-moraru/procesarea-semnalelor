import cv2
import numpy as np

img = cv2.imread('resources/poza.jpg')
print(type(img[:, :, 0].shape[0]))

cropped_rows, cropped_cols = 2048, 2048

with open('resources/poza_rgb.shape', "bw+") as f:
    f.write(cropped_rows.to_bytes(4, 'little'))
    f.write(cropped_cols.to_bytes(4, 'little'))

# TODO: crop to power of 2 rows, cols
rows, cols = img.shape[0], img.shape[1]
img_r = img[rows//2-1024:rows//2+1024, cols//2-1024:cols//2+1024, 0].copy().astype(dtype=np.uint8)
img_g = img[rows//2-1024:rows//2+1024, cols//2-1024:cols//2+1024, 1].copy().astype(dtype=np.uint8)
img_b = img[rows//2-1024:rows//2+1024, cols//2-1024:cols//2+1024, 2].copy().astype(dtype=np.uint8)

img_r.tofile('resources/poza_rgb.r')
img_g.tofile('resources/poza_rgb.g')
img_b.tofile('resources/poza_rgb.b')
