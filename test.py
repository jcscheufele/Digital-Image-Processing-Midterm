import cv2, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import choices

from skimage.metrics import structural_similarity as compare_ssim
import imutils

TRAIN_DATA = '../data/train'
TEST_DATA = '../data/test'

train_dirs = os.listdir(TRAIN_DATA)
test_dirs = os.listdir(TEST_DATA)

def get_next_positive_path():
    for locations in train_dirs:
        indv_loc_dir = os.listdir(TRAIN_DATA + "/" + locations)
        day = indv_loc_dir[0]
        night = indv_loc_dir[1]
        day_imgs = os.listdir(TRAIN_DATA + "/" + locations + "/" + day)
        night_imgs = os.listdir(TRAIN_DATA + "/" + locations + "/" + night)
        for dayimg in day_imgs:
            for nightimg in night_imgs:
                daypath = TRAIN_DATA + "/" + locations + "/" + day + "/" + dayimg
                nightpath = TRAIN_DATA + "/" + locations + "/" + night + "/" + nightimg
                yield daypath, nightpath

def get_next_negative_path():
    for locations in train_dirs:
        not_curr = set(train_dirs)-set([locations])
        indv_loc_dir = os.listdir(TRAIN_DATA + "/" + locations)
        day = indv_loc_dir[0]
        day_imgs = os.listdir(TRAIN_DATA + "/" + locations + "/" + day)
        for dayimg in day_imgs:
            daypath = TRAIN_DATA + "/" + locations + "/" + day + "/" + dayimg
            for local in not_curr:
                indv_loc_dir = os.listdir(TRAIN_DATA + "/" + local)
                night = indv_loc_dir[1]
                night_imgs = os.listdir(TRAIN_DATA + "/" + local + "/" + night)
                imgs = choices(night_imgs, k=5)
                for nightimg in imgs:
                    daypath = TRAIN_DATA + "/" + locations + "/" + day + "/" + dayimg
                    nightpath = TRAIN_DATA + "/" + local + "/" + night + "/" + nightimg
                    yield daypath, nightpath

'''pos_counter = 0
for img_paths in get_next_positive_path():
    print(pos_counter, img_paths, 1)
    pos_counter +=1
    img = cv2.imread(img_paths[0])
    img1 = cv2.imread(img_paths[1])
    print(img.shape, img1.shape)'''
 

'''neg_counter = 0
for img_paths in get_next_negative_path():
    print(neg_counter, img_paths, 0)
    neg_counter +=1'''


'''for dir in test_dirs:
    indv_dir = os.listdir(TEST_DATA + "/" + dir)
    for img in indv_dir:
        print(TEST_DATA + "/" + dir + "/" + img)'''


def similarityMatcher():
