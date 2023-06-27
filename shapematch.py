import cv2
import numpy as np
import matplotlib.pyplot as plt
def increase_contrast(image,sx,sy,thresh):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    high_pass = cv2.filter2D(image, -1, kernel)

    blurred = cv2.GaussianBlur(high_pass, (sx, sy), 3)
    _, final_image = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)

    return final_image
    
def givebb(image_path,mask_path):
	#Load Images
	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 
	
	#Apply Filters
	image = increase_contrast(image,7,7,70)
	mask = increase_contrast(mask,7,7,100)
	#cv2.imshow('test',image)
	#cv2.waitKey(0)

	# Contour detection
	contours1, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours2, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	mask=contours2[0]

	# Filtering contours
	filtered_contours = []
	for contour in contours1:
		area = cv2.contourArea(contour)
		x, y, w, h = cv2.boundingRect(contour)
		aspect_ratio = w / float(h)
		if area > 1000  and aspect_ratio > 1 and aspect_ratio < 4:
			filtered_contours.append(contour)
    		
	#print(len(filtered_contours))
	ret=[]
	for contour in filtered_contours:
		rt = cv2.matchShapes(contour,mask,3,0.0)
		x, y, w, h = cv2.boundingRect(contour)
	
		ret.append((rt,x,y,w,h))
		
		#cv2.drawContours(image, contour, -1, (0, 255, 0), 2)
	#print(ret)
	#_,x,y,w,h=ret[4]
	_,x,y,w,h=min(ret)
	print(_,x,y,w,h)
	cv2.rectangle(image, (x, y), (x + w, y + h), (100, 100, 200), 3)
	cv2.imshow('Result', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
givebb('test1.jpeg','temp.jpeg')

