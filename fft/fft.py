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

# fft
from numpy import fft
img_fft = fft.fftshift(fft.fft2(img_gray))
img_fft = np.abs(img_fft)
print("img_fft shape - {}".format(img_fft.shape))
plt.subplot(1, 4, 3)
plt.imshow(img_fft, cmap=plt.cm.gray)
plt.title('Fourier Magnitude Img')

# fft log
img_log_fft = np.log(img_fft)
print("img_log_fft shape - {}".format(img_log_fft.shape))
plt.subplot(1, 4, 4)
plt.imshow(img_log_fft, cmap=plt.cm.gray)
plt.title('Log Fourier Magnitude Img')

plt.show()