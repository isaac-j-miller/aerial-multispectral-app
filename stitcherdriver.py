# import the necessary packages
# this code was copied from https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
from stitcher import Stitcher
import argparse
import imutils
import cv2



# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
imageA = cv2.imread('piece1.png')
imageB = cv2.imread('piece2.png')
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)

# stitch the images together to create a panorama
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

# show the images
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)