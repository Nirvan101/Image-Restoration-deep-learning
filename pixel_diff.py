# Pixel diffusion 
# Make a mask of all the white regions
# and run cv2.inpaint using that mask

import cv2
import numpy as np

#load image
#path = '/home/nirvan/Documents/image_restor-ation_project/one/lol/in5.jpg'
path = '/home/nirvan/Documents/image_restoration_project/temp22.jpg'
img = cv2.imread(path)
img=np.array(img)

#select region
r = cv2.selectROI(img,False,False)
cropped = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

#cv2.imwrite('/home/nirvan/Documents/image_restoration_project/temp11.jpg',cropped)


#create mask
u = np.array([255,255,255],dtype='uint8')
l = np.array([150,150,150],dtype='uint8')

#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(cropped,l,u)

cv2.imwrite('/home/nirvan/Documents/image_restoration_project/mask.jpg',mask)


#apply mask
res = cv2.inpaint(cropped,mask,3,flags=cv2.INPAINT_TELEA)
cv2.imwrite('/home/nirvan/Documents/image_restoration_project/res.jpg',res)

#put the fixed region back in the original image'
img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = res
cv2.imwrite('/home/nirvan/Documents/image_restoration_project/temp22.jpg',img)