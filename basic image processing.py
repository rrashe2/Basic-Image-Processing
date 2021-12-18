# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt 


img = cv.imread("boat.png", cv.IMREAD_GRAYSCALE) 
print(img.shape) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = img.astype(np.float32)/255.

print(img[132,112])

cropped_image = img[0:300, 0:300]
cropped_image *= 255

cropped_image = cropped_image.astype(np.uint8)

cv.imshow("cropped", cropped_image)
cv.imwrite("Cropped imgae.jpg", cropped_image)
cv.waitKey(0)
cv.destroyAllWindows()

img = cv.imread("boat.png", cv.IMREAD_GRAYSCALE) 
Ma = 1./9.*np.array([[1.,1.,1.,],[1.,1.,1.],[1.,1.,1.,]])
from scipy.signal import convolve2d 
img_filtered = convolve2d(img, Ma) 
print(img_filtered.shape) 
plt.imshow(img_filtered, cmap='gray') 
plt.show() 
cv.imwrite("filteredimage.png", img_filtered)

img = cv.imread("boat.png", cv.IMREAD_GRAYSCALE) 
diff_filter = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
print(diff_filter)
img_diff_filtered = convolve2d(img, diff_filter) 
plt.imshow( img_diff_filtered, cmap='gray' ) 
plt.show()

sobel_x = np.array([-1, 2, -1])
horizontal_edge_detection = cv.filter2D(img, -1, sobel_x)
horizontal_edge_detection *= 255
horizontal_edge_detection = horizontal_edge_detection.astype(np.uint8)
cv.imshow("Horizontal edge detection image", horizontal_edge_detection)
cv.imwrite("Horizontal edge detection image.jpg", horizontal_edge_detection)
cv.waitKey(0)
cv.destroyAllWindows()

sobel_y = np.array([[-1], [2], [-1]])
vertical_edge_detection = cv.filter2D(img, -1, sobel_y)
vertical_edge_detection *= 255
vertical_edge_detection = vertical_edge_detection.astype(np.uint8)
cv.imshow("Vertical edge detection image", vertical_edge_detection)
cv.imwrite("Vertical edge detection image.jpg", vertical_edge_detection)
cv.waitKey(0)
cv.destroyAllWindows()

#BARBARA 

img_barbara = cv.imread("Barbara.jpg") 
img_barbara = cv.cvtColor(img_barbara, cv.COLOR_BGR2RGB)
print(img_barbara.shape) 
plt.imshow(img_barbara, cmap='gray') 
plt.show() 

img_barbara = img_barbara.astype(np.float32)/255.

print(img_barbara[132,112])

cropped_barbara_image = img_barbara[0:300, 0:300]
cropped_barbara_image *= 255

cropped_barbara_image = cropped_barbara_image.astype(np.uint8)
cropped_barbara_image = cv.cvtColor(cropped_barbara_image, cv.COLOR_RGB2BGR)
cv.imshow("cropped barbara", cropped_barbara_image)
cv.imwrite("Cropped barbara imgae.jpg", cropped_barbara_image)
cv.waitKey(0)
cv.destroyAllWindows()

img_barbara = cv.imread("Barbara.jpg", cv.IMREAD_GRAYSCALE)
print(img_barbara.shape)
Ma = 1./9.*np.array([[1.,1.,1.,],[1.,1.,1.],[1.,1.,1.,]])
from scipy.signal import convolve2d 
img_filtered_barbara = convolve2d(img_barbara, Ma) 
print(img_filtered_barbara.shape) 
#img_filtered_barbara = cv.cvtColor(img_filtered_barbara, cv.COLOR_RGB2BGR)
plt.imshow(img_filtered_barbara, cmap='gray') 
plt.show() 
cv.imwrite("filteredimagebarbara.jpg", img_filtered_barbara)


img_barbara = cv.imread("Barbara.jpg", cv.IMREAD_GRAYSCALE) 
diff_filter = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
print(diff_filter)
img_diff_filtered_barbara = convolve2d(img_barbara, diff_filter)
plt.imshow( img_diff_filtered_barbara, cmap='gray' ) 
plt.show()

sobel_x = np.array([-1, 2, -1])
horizontal_edge_detection_barbara = cv.filter2D(img_barbara, -1, sobel_x)
horizontal_edge_detection_barbara *= 255
horizontal_edge_detection_barbara = horizontal_edge_detection_barbara.astype(np.uint8)
cv.imshow("Horizontal edge detection barbara image", horizontal_edge_detection_barbara)
cv.imwrite("Horizontal edge detection barbara image.jpg", horizontal_edge_detection_barbara)
cv.waitKey(0)
cv.destroyAllWindows()

sobel_y = np.array([[-1], [2], [-1]])
vertical_edge_detection_barbara = cv.filter2D(img_barbara, -1, sobel_y)
vertical_edge_detection_barbara *= 255
vertical_edge_detection_barbara = vertical_edge_detection_barbara.astype(np.uint8)
cv.imshow("Vertical edge detection barbara image", vertical_edge_detection_barbara)
cv.imwrite("Vertical edge detection barbara image.jpg", vertical_edge_detection_barbara)
cv.waitKey(0)
cv.destroyAllWindows()
