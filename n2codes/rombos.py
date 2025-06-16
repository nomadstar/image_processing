import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.transform import hough_line, hough_line_peaks

# Load the image
img = cv2.imread('imagenes/rombo.png', cv2.IMREAD_GRAYSCALE)
BW = cv2.Canny(img, 50, 150, apertureSize=3)
print("BW shape:", BW.shape)
angulos = np.linspace(-np.pi / 2, np.pi / 2, 360)

h, theta, d = hough_line(BW, theta=angulos)

max_values= hough_line_peaks(h, theta, d)

for accum, theta, rho in zip(*max_values):
    print(f"Accum: {accum}, Theta: {np.rad2deg(theta)}, Rho: {rho}")

# Solemne 2 jueves 26 de este mes
#Display the image
plt.imshow(np.log(h + 1), cmap='turbo_r', extent=(-np.rad2deg(angulos[0]), -np.rad2deg(angulos[-1]), d[-1], d[0]), aspect='auto')
plt.axis('on')
plt.show()

