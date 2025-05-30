import cv2
import numpy as np

import matplotlib.pyplot as plt 

def loadimage(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dst = np.zeros_like(gray, dtype='float32')
    gray = cv2.normalize(gray.astype('float32'), dst, 0.0, 1.0, cv2.NORM_MINMAX)
    return gray

def showimage(img, title=''):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def starspectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    star_spectrum = np.log(np.abs(fshift) + 1)
    return star_spectrum

def turbulentNoise(img):
    x,y = img.shape
    X, Y = np.meshgrid(np.linspace(-0.5, 0.5, y), np.linspace(-0.5, 0.5, x))
    k = 10  # turbulence strength parameter, adjust as needed
    equation = np.exp(-k * (X**2 + Y**2)**(5/6))


if __name__ == "__main__":
    img = loadimage('imagenes/pato.jpg')
    showimage(img, 'Original Image')
    
    # Show the spectrum of the image
    star_spectrum = starspectrum(img)
    showimage(star_spectrum, 'Star Spectrum')
    
    # Apply turbulence noise
    noise_img = turbulentNoise(img)
    showimage(noise_img, 'Turbulent Noise Image')