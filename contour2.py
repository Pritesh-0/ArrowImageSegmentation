import cv2
import numpy as np

# Load the image
image = cv2.imread('test2.jpeg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Thresholding
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#binary=cv2.bitwise_not(binary)
cv2.imshow('binary',binary)
cv2.waitKey(0)

# Contour detection
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filtering contours
filtered_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    
    # Adjust the following conditions as per your requirements
    if area > 2000 and area <4000 and aspect_ratio > 1 and aspect_ratio < 4:
        filtered_contours.append(contour)
print(len(filtered_contours))
# Identify the arrow
for contour in filtered_contours:
    # Add your arrow detection logic here
    # You can analyze the contour's shape, symmetry, directionality, etc.

# Visualize the result
    cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break

