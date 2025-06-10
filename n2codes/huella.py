from PIL import Image
from PIL import ImageFilter
from cv2 import mean
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Cargar la imagen
img = Image.open('imagenes/huella.png')

img_gray = img.convert('L')
pixel_values = np.array(img_gray).flatten()
# Normalizar los valores de píxel al rango [0, 1]
normalized_pixels = pixel_values / 255.0

plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

plt.figure()
plt.hist(normalized_pixels, bins=256, range=(0, 1), color='green')
plt.title('Histograma de la imagen normalizada')
plt.xlabel('Valor de píxel normalizado')
plt.ylabel('Frecuencia')
plt.show()

# Calcular el umbral de Otsu sin usar librerías externas

hist = [0] * 256
for value in pixel_values:
    hist[value] += 1

total = len(pixel_values)
sum_total = sum(i * hist[i] for i in range(256))

sumB = 0
wB = 0
max_var = 0
threshold = 0

# Preparar la figura para la animación
fig, ax = plt.subplots()
bars = ax.bar(range(256), hist, color='green', alpha=0.6)
threshold_line, = ax.plot([0, 0], [0, max(hist)], 'r-', linewidth=2)
ax.set_xlim(0, 255)
ax.set_ylim(0, max(hist) * 1.1)
ax.set_title('Iteraciones de Otsu en el histograma')
ax.set_xlabel('Valor de píxel')
ax.set_ylabel('Frecuencia')

# Para la animación
scatter = None
lines = []
var_between_list = []
sum_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))


def animate(t):
    global sumB, wB, max_var, threshold, scatter, lines
    if t == 0:
        sumB = 0
        wB = 0
        max_var = 0
        threshold = 0
        var_between_list.clear()
        for line in lines:
            line.remove()
        lines = []

    wB = sum(hist[:t+1])
    if wB == 0 or wB == total:
        threshold_line.set_xdata([t, t])
        sum_text.set_text('')
        return bars
    wF = total - wB
    sumB = sum(i * hist[i] for i in range(t+1))
    mB = sumB / wB
    mF = (sum_total - sumB) / wF
    var_between = wB * wF * (mB - mF) ** 2
    var_between_list.append(var_between)
    if var_between > max_var:
        max_var = var_between
        threshold = t

    threshold_line.set_xdata([t, t])

    for i, bar in enumerate(bars):
        if i <= t:
            bar.set_color('orange')
        else:
            bar.set_color('green')

    if scatter is not None:
        scatter.remove()
    scatter = ax.scatter(
        range(len(var_between_list)),
        [v * max(hist) / max(var_between_list) for v in var_between_list],
        color='blue', s=5, label='Varianza entre clases'
    )

    # Mostrar la sumatoria en la animación
    sum_text.set_text(f'Sumatoria: {sumB:.0f}')

    if t == 255:
        min_idx = int(np.argmin(var_between_list))
        max_idx = int(np.argmax(var_between_list))
        min_line = ax.axvline(min_idx, color='purple', linestyle='--', linewidth=2, label='Varianza mínima')
        max_line = ax.axvline(max_idx, color='red', linestyle='--', linewidth=2, label='Varianza máxima')
        lines.extend([min_line, max_line])
        ax.legend()

    return list(bars) + [threshold_line, scatter, sum_text] + lines

ani = animation.FuncAnimation(fig, animate, frames=256, interval=20, blit=False, repeat=False)
plt.show()

print(f"Umbral de Otsu calculado: {threshold}")

# Binarizar la imagen usando el umbral calculado
binary_img = img_gray.point(lambda p: 255 if p > threshold else 0)
plt.figure()
plt.imshow(binary_img, cmap='gray')
plt.axis('off')
plt.title('Imagen binarizada (Otsu)')
plt.show()
