import torch
import cv2
import numpy as np
import datetime
#import pytz
#import pywt

from torchvision.io import ImageReadMode
from torchvision.io import read_image
from torchvision import transforms

from torch.utils.data import Dataset
from torch import as_tensor, cat, flatten, div
from matplotlib import pyplot as plt

class BasicDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __len__(self):
        return len(self.y)
        
    def getImage(self):
        pass

    def __getitem__(self, idx):
        return self.X[idx], as_tensor(self.y[idx], dtype=torch.float32)
