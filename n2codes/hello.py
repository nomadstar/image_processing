from locale import normalize
from re import sub
import cv2
from matplotlib.colors import Normalize
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt 

A = cv2.imread('imagenes/rice.png', 0)
if A is None:
    raise FileNotFoundError("Image 'imagenes/rice.png' not found. Please check the path.")

q = (13, 13)
images = []
B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, q)
des = cv2.erode(A, B, iterations=1)
res = cv2.dilate(des, B, iterations=1)
normalized = cv2.normalize(np.int8(res), None, 0, 255, cv2.NORM_MINMAX)
des = cv2.erode(res, B, iterations=1)
subtract = cv2.subtract(A, normalized)
images.append((A, res, des, normalized, subtract))

# plot images
fig, ax = plt.subplots(1, 5, figsize=(15, 3))
titles = ['Original Image', 'Dilated Image', 'Eroded Image', 'Median Blurred Image', 'Result Image']
for j in range(5):
    ax[j].imshow(images[0][j], cmap='gray')
    ax[j].set_title(titles[j])
    ax[j].axis('off')
plt.tight_layout()
plt.show()
