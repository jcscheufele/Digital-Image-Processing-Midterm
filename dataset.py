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

def createTransform():
    p144 = (144, 256)
    p240 = (240, 426)
    p360 = (360, 640)
    p480 = (480, 848)
    crop = (355, 644)
    listOfTransforms = [
        transforms.Resize(p240)
        ]
    return transforms.Compose(listOfTransforms)

class BasicDataset(Dataset):
    def __init__(self):
        self.X = []
        self.y = []
        self.transform = createTransform()
        self.makeXnY()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def makeXnY(self):
        print("Making Positives...")
        counter = 0
        for img_paths in get_next_positive_path():
            print(f"img pair {counter}")
            img1 = read_image(img_paths[0])
            img2 = read_image(img_paths[1])

            img1 = self.transform(img1)
            img2 = self.transform(img2)

            self.X.append(img1, img2)
            self.y.append(1)
            counter += 1

        print("Making Negatives...")
        for img_paths in get_next_negative_path():
            print(f"img pair {counter}")
            img1 = read_image(img_paths[0])
            img2 = read_image(img_paths[1])

            img1 = self.transform(img1)
            img2 = self.transform(img2)

            self.X.append(img1, img2)
            self.y.append(0)
            counter += 1
        print("Completed")

