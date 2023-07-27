import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 14), dpi=100)

# read image
img = cv2.imread("lena.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("img_rgb shape - {}".format(img_rgb.shape))
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("RGB Img")

# convert to gray
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
print("img_gray shape - {}".format(img_gray.shape))
plt.subplot(1, 3, 2)
plt.imshow(img_gray, cmap = plt.cm.gray)
plt.title('Gray Img')

# hog
from skimage import feature as ft
features = ft.hog(img_gray, orientations=6, pixels_per_cell=[10, 10], cells_per_block=[3, 3], visualize=True)
print("img_hog shape - {}".format(features[1].shape))
plt.subplot(1, 3, 3)
plt.imshow(features[1], plt.cm.gray)
plt.title('HOG Img')

plt.show()
