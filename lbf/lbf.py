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

# lbf
from skimage.feature import local_binary_pattern

radius = 1
n_points = 8

lbp = local_binary_pattern(img_gray, n_points, radius)
print("img_lbp shape - {}".format(lbp.shape))
plt.subplot(1, 3, 3)
plt.imshow(lbp, plt.cm.gray)
plt.title('LBF Img')

plt.show()