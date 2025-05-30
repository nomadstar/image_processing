import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mpimg

def loadimage(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return gray

def showimage(img, title=''):
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis('off')
    plt.show()

def shownoise(img):

    fft = np.fft.fft2(img)
    fftshift = np.fft.fftshift(fft)
    plt.figure(figsize=(50, 50))
    plt.imshow(np.log(np.abs(fftshift)), cmap='gray')
    plt.title('Espectro de la imagen')
    plt.axis('off')
    plt.show()

def makeHbtw_pts(h,k,a,b,n,m):
    m, n = img.shape
    x = np.linspace(-n/2, n/2, n)
    y = np.linspace(-m/2, m/2, m)
    X, Y = np.meshgrid(x, y)
    H = 1 - ((X - h)**2 / (a**2) + (Y - k)**2 / (b**2))
    H[H < 0] = 0
    return H


def cleanoise(img):
    m, n = img.shape
    # PASO 1)
    # determinamos el espectro de la imagen
    F = np.fft.fft2(img)
    FS = np.fft.fftshift(F)
    # creamos un filtro parabanda
    # la ecuacion es una parabola de segundo grado donde la ecuacion 1 = (x)^2/(16^2)+(y)^2/(1^2) y de grosor 2
    H = makeHbtw_pts(0, 0, 200, 100, n, m)
    NH = makeHbtw_pts(0, 0, 300, 50, n, m)
    # suma H y NH
    H = H - NH
    # aplicamos el filtro a la imagen
    plt.figure(figsize=(50, 50))
    plt.imshow(H, cmap='gray')

    plt.title('Filtro parabanda')
    plt.axis('off')
    plt.show()
    

if __name__=='__main__':
    # Load and display the image
    img = loadimage('imagenes/pato.jpg')
    showimage(img, 'Pato - Escala de Grises')
    # Show the noise in the image
    shownoise(img)
    # Clean the noise in the image
    cleanoise(img)


