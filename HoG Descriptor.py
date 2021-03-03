import cv2
from skimage import feature
import matplotlib.pyplot as plt


image_dir='flower.jpg'
image = cv2.imread(image_dir)

(hog, hog_image) = feature.hog(image, orientations=9,
                    pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                    block_norm='L2-Hys', visualize=True, transform_sqrt=True)

plt.close()
plt.subplot(121)
cv2.imshow('Original Image', image)
plt.subplot(122)
cv2.imshow('HOG Image', hog_image)
cv2.waitKey(0)