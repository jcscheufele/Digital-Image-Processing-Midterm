import torch
import datetime
#import pytz
#import pywt

from torchvision.io import ImageReadMode
from torchvision.io import read_image
from torchvision import transforms

from torch.utils.data import Dataset
from torch import as_tensor, cat, flatten, div
from matplotlib import pyplot as plt
import os

import cv2, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import choices, choice

TRAIN_DATA = '../data/train'
TEST_DATA = '../data/test'
VALID_DATA = '../data/valid'

train_dirs = os.listdir(TRAIN_DATA)
test_dirs = os.listdir(TEST_DATA)
valid_dirs = os.listdir(VALID_DATA)

def tr_get_next_positive_path():
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

def tr_get_next_negative_path():
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

def te_get_next_positive_path():
    for locations in test_dirs:
        indv_loc_dir = os.listdir(TEST_DATA + "/" + locations)
        day = indv_loc_dir[0]
        night = indv_loc_dir[1]
        day_imgs = os.listdir(TEST_DATA + "/" + locations + "/" + day)
        night_imgs = os.listdir(TEST_DATA + "/" + locations + "/" + night)
        for dayimg in day_imgs:
            for nightimg in night_imgs:
                daypath = TEST_DATA + "/" + locations + "/" + day + "/" + dayimg
                nightpath = TEST_DATA + "/" + locations + "/" + night + "/" + nightimg
                yield daypath, nightpath

def te_get_next_negative_path():
    for locations in test_dirs:
        not_curr = set(test_dirs)-set([locations])
        indv_loc_dir = os.listdir(TEST_DATA + "/" + locations)
        day = indv_loc_dir[0]
        day_imgs = os.listdir(TEST_DATA + "/" + locations + "/" + day)
        for dayimg in day_imgs:
            daypath = TEST_DATA + "/" + locations + "/" + day + "/" + dayimg
            for local in not_curr:
                indv_loc_dir = os.listdir(TEST_DATA + "/" + local)
                night = indv_loc_dir[1]
                night_imgs = os.listdir(TEST_DATA + "/" + local + "/" + night)
                imgs = choices(night_imgs, k=5)
                for nightimg in imgs:
                    daypath = TEST_DATA + "/" + locations + "/" + day + "/" + dayimg
                    nightpath = TEST_DATA + "/" + local + "/" + night + "/" + nightimg
                    yield daypath, nightpath

def va_get_next_positive_path():
    for locations in valid_dirs:
        indv_loc_dir = os.listdir(VALID_DATA + "/" + locations)
        day = indv_loc_dir[0]
        night = indv_loc_dir[1]
        day_imgs = os.listdir(VALID_DATA + "/" + locations + "/" + day)
        night_imgs = os.listdir(VALID_DATA + "/" + locations + "/" + night)
        for dayimg in day_imgs:
            for nightimg in night_imgs:
                daypath = VALID_DATA + "/" + locations + "/" + day + "/" + dayimg
                nightpath = VALID_DATA + "/" + locations + "/" + night + "/" + nightimg
                yield daypath, nightpath

def va_get_next_negative_path():
    for locations in valid_dirs:
        not_curr = set(valid_dirs)-set([locations])
        indv_loc_dir = os.listdir(VALID_DATA + "/" + locations)
        day = indv_loc_dir[0]
        day_imgs = os.listdir(VALID_DATA + "/" + locations + "/" + day)
        for dayimg in day_imgs:
            daypath = VALID_DATA + "/" + locations + "/" + day + "/" + dayimg
            for local in not_curr:
                indv_loc_dir = os.listdir(VALID_DATA + "/" + local)
                night = indv_loc_dir[1]
                night_imgs = os.listdir(VALID_DATA + "/" + local + "/" + night)
                imgs = choices(night_imgs, k=5)
                for nightimg in imgs:
                    daypath = VALID_DATA + "/" + locations + "/" + day + "/" + dayimg
                    nightpath = VALID_DATA + "/" + local + "/" + night + "/" + nightimg
                    yield daypath, nightpath

def createTransform():
    p144 = (144, 256)
    p240 = (240, 426)
    p360 = (360, 640)
    p480 = (480, 848)
    crop = (355, 644)
    listOfTransforms = [
        transforms.Resize(p144)
        ]
    return transforms.Compose(listOfTransforms)

class BasicDataset(Dataset):
    def __init__(self, train):
        self.train = train
        self.X_day = []
        self.X_night = []
        self.y = []
        self.transform = createTransform()
        self.makeXnY()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_day[idx], self.X_night[idx], as_tensor(self.y[idx], dtype=torch.float32)

    def makeXnY(self):
        print("Making Positives...")
        counter = 0
        if self.train == "Train":
            pos_gen = tr_get_next_positive_path
            neg_gen = tr_get_next_negative_path
        elif self.train == "Test":
            pos_gen = te_get_next_positive_path
            neg_gen = te_get_next_negative_path
        elif self.train == "Valid":
            pos_gen = va_get_next_positive_path
            neg_gen = va_get_next_negative_path
        for img_paths in pos_gen():
            print(f"img pair {counter}", end="\r")
            day = read_image(img_paths[0], mode=ImageReadMode.GRAY)
            night = read_image(img_paths[1], mode=ImageReadMode.GRAY)

            day = self.transform(day)
            night = self.transform(night)

            day = div(day, 255)
            night = div(night, 255)

            self.X_day.append(day)
            self.X_night.append(night)
            self.y.append(1.0)
            counter += 1

        print("Making Negatives...")
        for img_paths in neg_gen():
            print(f"img pair {counter}", end="\r")
            day = read_image(img_paths[0], mode=ImageReadMode.GRAY)
            night = read_image(img_paths[1], mode=ImageReadMode.GRAY)

            day = self.transform(day)
            night = self.transform(night)

            day = div(day, 255)
            night = div(night, 255)

            self.X_day.append(day)
            self.X_night.append(night)
            self.y.append(0.0)
            counter += 1
        print("Completed")

