# Based on 61_A_notch_filter.py
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mpimg

def turbulentNoise(img, k=10 ,a=4, b=2, T=1):
    """
    Generate turbulent noise based on the input image.
    
    Parameters:
    img (numpy.ndarray): Input image.
    k (float): Turbulence strength parameter.
    
    Returns:
    numpy.ndarray: Image with turbulent noise applied.
    """
    x, y = img.shape
    # Create frequency coordinates u and v
    u = np.linspace(-0.5, 0.5, y)
    v = np.linspace(-0.5, 0.5, x)
    U, V = np.meshgrid(u, v)
    # Compute the argument for the formula
    arg = np.pi * (a * U + b * V)
    # Compute the turbulent noise formula
    formulae = T * np.sinc(arg / np.pi) * np.exp(1j * k * arg)
    return formulae

def loadimage(filename):
    img = mpimg.imread(filename)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def ftimage(img):
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    return Fshift

def ftshiftimage(img):
    Fshift = np.fft.fftshift(img)
    return Fshift

def desapplyfourier(img):
    F_ishift = np.fft.ifftshift(img)
    return np.fft.ifft2(F_ishift)

def showimage(img, title=''):
    plt.imshow(np.abs(img), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def starspectrum(img):
    """
    Compute the star spectrum of the image.
    
    Parameters:
    img (numpy.ndarray): Input image.
    
    Returns:
    numpy.ndarray: Logarithmic spectrum of the image.
    """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    star_spectrum = np.log(np.abs(fshift) + 1)
    return star_spectrum

if __name__ == "__main__":
    img = loadimage('imagenes/pato.jpg')
    showimage(img, 'Original Image')
        
    # Show the spectrum of the image
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    axs = axs.flatten()
    
    star_spectrum = starspectrum(img)
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(star_spectrum, cmap='gray')
    axs[1].set_title('Star Spectrum')
    axs[1].axis('off')

    noisestar =star_spectrum*turbulentNoise(star_spectrum, k=10)
    axs[2].imshow(np.abs(noisestar), cmap='gray')
    axs[2].set_title('Turbulent Noise on Star Spectrum')
    axs[2].axis('off')

    noisestar=ftshiftimage(noisestar)
    axs[3].imshow(np.abs(noisestar), cmap='gray')
    axs[3].set_title('Shifted Turbulent Noise')
    axs[3].axis('off')

    # Apply inverse Fourier transform to the noisy spectrum
    noisy_image = np.fft.ifft2(np.fft.ifftshift(noisestar))
    axs[4].imshow(np.abs(noisy_image), cmap='gray')
    axs[4].set_title('Inverse Fourier Transform of Noisy Spectrum')
    axs[4].axis('off')

    # Fill remaining subplots with empty images or remove them
    for i in range(5, 9):
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

     


