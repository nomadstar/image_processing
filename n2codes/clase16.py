import cv2
import matplotlib.pyplot as plt

A = cv2.imread('imagenes/cars.jpg',0)

B = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
print(B)


res = cv2.dilate(A, B, iterations=1)
des = cv2.erode(res, B, iterations=1)
C = cv2.bitwise_not(des)
R = cv2.bitwise_and(A, C)


fig, ax = plt.subplots(1, 5, figsize=(15, 5))
ax[0].imshow(A, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(res, cmap='gray')
ax[1].set_title('Dilated Image')
ax[1].axis('off')
ax[2].imshow(des, cmap='gray')
ax[2].set_title('Eroded Image')
ax[2].axis('off')
ax[3].imshow(C, cmap='gray')
ax[3].set_title('Complement Image')
ax[3].axis('off')
ax[4].imshow(R, cmap='gray')
ax[4].set_title('Result Image')
ax[4].axis('off')
plt.tight_layout()
plt.show()
