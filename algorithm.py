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

day = cv2.imread('../data/train/00000850/Day/20151101_142506.jpg')
night = cv2.imread('../data/train/00000850/Night/20151101_072507.jpg')
h = 144
w = 256

re = (w,h)
day = cv2.resize(day ,re, interpolation = cv2.INTER_AREA)
night = cv2.resize(night ,re, interpolation = cv2.INTER_AREA)

# convert the images to grayscale
day_g = cv2.cvtColor(day, cv2.COLOR_BGR2GRAY)
night_g = cv2.cvtColor(night, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(day_g, night_g, full=True)
#(score,diff)=compare_ssim(grayA, grayB, multichannel=True, gaussian_weights=True, sigma=2, use_sample_covariance=False, data_range=255,full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))


thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(day, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(night, (x, y), (x + w, y + h), (0, 0, 255), 2)
# show the output images

cv2.imwrite("/testImgs/day.png", day)
cv2.imwrite("/testImgs/night.png", night)
cv2.imwrite("/testImgs/diff.png", diff)
cv2.imwrite("/testImgs/thresh.png", thresh)
cv2.waitKey(0)