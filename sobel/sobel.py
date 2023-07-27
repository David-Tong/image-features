import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 14), dpi=100)

# read image
img = cv2.imread("lena.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("img_rgb shape - {}".format(img_rgb.shape))
plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("RGB Img")

# convert to gray
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
print("img_gray shape - {}".format(img_gray.shape))
plt.subplot(1, 4, 2)
plt.imshow(img_gray, cmap=plt.cm.gray)
plt.title('Gray Img')

# sobel
sobelx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=21)
sobely = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=21)
img_sobel = np.sqrt(sobelx**2.0 + sobely**2.0)
img_sobel = img_sobel / np.max(img_sobel)
print("img_sobel shape - {}".format(img_sobel.shape))
plt.subplot(1, 4, 3)
plt.imshow(img_sobel, cmap=plt.cm.gray)
plt.title('Sobel Img')

# canny
img_canny = cv2.Canny(image=(img_sobel * 255).astype(np.uint8), threshold1=0, threshold2=100)
img_canny = img_canny / np.max(img_canny)
print("img_canny shape - {}".format(img_canny.shape))
plt.subplot(1, 4, 4)
plt.imshow(img_canny, cmap=plt.cm.gray)
plt.title('Canny Img')

plt.show()
