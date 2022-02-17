import cv2, os
import pandas as pd

TRAIN_DATA = '../data/train'
TEST_DATA = '../data/test'

train_dirs = os.listdir(TRAIN_DATA)
test_dirs = os.listdir(TEST_DATA)
print(train_dirs, test_dirs)

for dir in train_dirs:
    indv_dir = os.listdir(TRAIN_DATA + "/" + dir)
    for img in indv_dir:
        print(TRAIN_DATA + "/" + dir + "/" + img)

for dir in test_dirs:
    indv_dir = os.listdir(TEST_DATA + "/" + dir)
    for img in indv_dir:
        print(TEST_DATA + "/" + dir + "/" + img)



img = cv2.imread("../data/train/00000850/20151101_072507.jpg")
print(img.shape)

scale_percent = 25 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
print(resized.shape)
#dehaze(img)
cv2.imwrite("first.jpg", img)
cv2.imwrite("enhanced.jpg", resized)