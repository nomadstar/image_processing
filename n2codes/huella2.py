from PIL import Image

import matplotlib.pyplot as plt

# Cargar la imagen
try:
    img = Image.open('imagenes/huella.png')
except:
    print("Error: No se pudo cargar la imagen 'huella.png'. Asegúrate de que el archivo exista en el directorio actual.")
    exit()
# Mostrar la imagen
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

# Mostrar histograma de la imagen en escala de grises
img_gray = img.convert('L')
plt.figure()
plt.hist(list(img_gray.getdata()), bins=256, range=(0, 255), color='gray')
plt.title('Histograma de la imagen')
plt.xlabel('Valor de píxel')
plt.ylabel('Frecuencia')
plt.show()