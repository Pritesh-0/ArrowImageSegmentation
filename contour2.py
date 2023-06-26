import cv2
import numpy as np
import matplotlib.pyplot as plt
def increase_contrast(image):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    high_pass = cv2.filter2D(image, -1, kernel)

    blurred = cv2.GaussianBlur(high_pass, (7, 7), 3)
    _, final_image = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)

    return final_image
    

# Load the image
image_path = 'test3.jpeg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
temp = cv2.imread('temp.jpeg', cv2.IMREAD_GRAYSCALE) 
temp1 = cv2.imread('temp1.jpeg', cv2.IMREAD_GRAYSCALE) 
image = increase_contrast(image)
temp = increase_contrast(temp)
temp1 = increase_contrast(temp1)



# Thresholding
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
_, tempbin = cv2.threshold(temp1, 127, 255, cv2.THRESH_BINARY)


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
    if area > 1000  and aspect_ratio > 1 and aspect_ratio < 4:
        filtered_contours.append(contour)
print(len(filtered_contours))
ret=[]
for contour in filtered_contours:
	rt = cv2.matchShapes(contour,mask,1,0.0)
	x, y, w, h = cv2.boundingRect(contour)
	
	ret.append((rt,x,y,w,h))
	#cv2.drawContours(image, contour, -1, (0, 255, 0), 2)
print(ret)
#_,x,y,w,h=ret[4]
_,x,y,w,h=min(ret)
print(_,w,h)
cv2.rectangle(image, (x, y), (x + w, y + h), (100, 100, 200), 3)
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

