import cv2 
import numpy as np 
import argparse
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required = True, help = "path to the image")

args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray , 225 , 255, cv2.THRESH_BINARY_INV)[1]

cv2.imshow('Original', image)
cv2.imshow('Thresholding', thresh)

cnts = cv2.findContours(thresh , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

hullImage = np.zeros(gray.shape[:2], dtype = "uint8")

for (i, c) in enumerate(cnts):
    area = cv2.contourArea(c)
    (x, y, w, h) = cv2.boundingRect(c)

    #Aspect Ratio
    aspectRatio = w / float(h)

    #Extent
    extent = area / float(w * h)

    #Convex Hull
    hull = cv2.convexHull(c)
    hullArea = cv2.contourArea(hull)

    #Solidity
    solidity = area / float(hullArea)

    cv2.drawContours(hullImage , [hull], -1, 255, -1)
    cv2.drawContours(image , [c], -1, (228, 0, 159), 3)

    shape = ""

    if aspectRatio >= 0.98 and aspectRatio <= 1.02:
        shape = "SQUARE"
    elif aspectRatio >= 3.0:
        shape = "RECTANGLE"
    elif extent < 0.65:
        shape = "L-PIECE"
    elif solidity > 0.80:
        shape = "Z-PIECE"

    cv2.putText(image , shape , (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (228, 0, 159), 1)

    print("Contour: {} Aspect Ratio={:.2f} Extent={:.2f} Soidity={:.2f}".format(i+1, aspectRatio, extent, solidity))
    
    cv2.imshow('Image', image)
    cv2.imshow('Convex Hull', hullImage)
    cv2.waitKey(0)
cv2.waitKey(0)