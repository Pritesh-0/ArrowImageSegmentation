import cv2
import numpy as np
import matplotlib.pyplot as plt
def increase_contrast(image):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    high_pass = cv2.filter2D(image, -1, kernel)
    #contrast = cv2.subtract(image, high_pass)

    blurred = cv2.GaussianBlur(high_pass, (9, 9), 6)
    _, final_image = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

    return final_image
# Load the image
image_path = 'test2.jpeg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
temp = cv2.imread('temp.jpeg', cv2.IMREAD_GRAYSCALE) 
image = increase_contrast(image)
# Convert to grayscale
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Thresholding
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
_, tempbin = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
 
#binary=cv2.bitwise_not(binary)

# Contour detection
contours1, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(tempbin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
mask=contours2[0]

# Filtering contours
filtered_contours = []
for contour in contours1:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    
    # Adjust the following conditions as per your requirements
    if area > 1000 and area<8000  and aspect_ratio > 1 and aspect_ratio < 4:
        filtered_contours.append(contour)
print(len(filtered_contours))
# Identify the arrow
ret=[]
for contour in filtered_contours:
	rt = cv2.matchShapes(contour,mask,1,0.0)
	x, y, w, h = cv2.boundingRect(contour)
	
	ret.append((rt,x,y,w,h))
	#cv2.drawContours(image, contour, -1, (0, 255, 0), 2)
print(ret)
_,x,y,w,h=min(ret)
cv2.rectangle(image, (x, y), (x + w, y + h), (200, 0, 0), 2)
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

