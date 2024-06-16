from scipy import ndimage
from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt

image = plt.imread("1.jpeg")
print(image.shape)

gray = rgb2gray(image)
cv2.imshow('img', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# sobel_horizontal = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([-1, -2, -1])])
# sobel_vertical = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]), np.array([-1, 0, 1])])
kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])

# out_h = ndimage.convolve(gray, sobel_horizontal, mode='reflect')
# out_v = ndimage.convolve(gray, sobel_vertical, mode='reflect')

out_l = ndimage.convolve(gray, kernel_laplace, mode='reflect')
plt.imshow(out_l, cmap='gray')
plt.show()
