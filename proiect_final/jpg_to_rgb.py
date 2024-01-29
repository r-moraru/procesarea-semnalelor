import cv2
import numpy as np

img = cv2.imread('resources/poza.JPG')
print(type(img[:, :, 0].shape[0]))

cropped_rows, cropped_cols = 2048, 2048

with open('resources/img.shape', "bw+") as f:
    f.write(cropped_rows.to_bytes(4, 'little'))
    f.write(cropped_cols.to_bytes(4, 'little'))

rows, cols = img.shape[0], img.shape[1]

rows_start = rows//2 - cropped_rows//2
rows_end = rows//2 + cropped_rows//2
cols_start = cols//2 - cropped_cols//2
cols_end = cols//2 + cropped_cols//2

img_r = img[rows_start:rows_end, cols_start:cols_end, 0].copy().astype(dtype=np.uint8)
img_g = img[rows_start:rows_end, cols_start:cols_end, 1].copy().astype(dtype=np.uint8)
img_b = img[rows_start:rows_end, cols_start:cols_end, 2].copy().astype(dtype=np.uint8)

img_r.tofile('resources/img_r.mat')
img_g.tofile('resources/img_g.mat')
img_b.tofile('resources/img_b.mat')
