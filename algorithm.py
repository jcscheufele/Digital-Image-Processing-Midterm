#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 20:32:23 2022

@author: JoseFigueroa
"""

from skimage.metrics import structural_similarity as compare_ssim
import imutils
import cv2
import matplotlib as plt

imageA = cv2.imread('/Users/JoseFigueroa/Documents/WPI/(2)SeconSemesterECE/ECE 545 Digital Image Processing/Project/MidtermProject/nighttime place recognition dataset/test/00021510/20151102_103048.jpg')
imageB1 = cv2.imread('/Users/JoseFigueroa/Documents/WPI/(2)SeconSemesterECE/ECE 545 Digital Image Processing/Project/MidtermProject/nighttime place recognition dataset/train/00004368/20151102_083445.jpg')
h=555
w=800

re=(w,h)

imageB=cv2.resize(imageB1,re, interpolation = cv2.INTER_AREA)

# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(grayA, grayB, full=True)
#(score,diff)=compare_ssim(grayA, grayB, multichannel=True, gaussian_weights=True, sigma=2, use_sample_covariance=False, data_range=255,full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))


thresh = cv2.threshold(diff, 0, 255, 
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)


# loop over the contours
for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
# show the output images

cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)