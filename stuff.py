import os
import cv2 as cv 
from random import choices

TRAIN_DATA = '../data/train'
TEST_DATA = '../data/test'
VALID_DATA = '../data/valid'

train_dirs = os.listdir(TRAIN_DATA)
test_dirs = os.listdir(TEST_DATA)
valid_dirs = os.listdir(VALID_DATA)

'''lengths = []

for location in train_dirs:
    times = os.listdir(TRAIN_DATA + "/" + location)
    days = os.listdir(TRAIN_DATA + "/" + location + "/" + times[0])
    nights = os.listdir(TRAIN_DATA + "/" + location + "/" + times[1])
    #lengths.append([len(days), len(nights)])
    #print("Train", location, len(days), len(nights))
    for day, night in zip(days, nights):
        day = cv.imread(TRAIN_DATA + "/" + location + "/" + times[0] + "/" + day, 0)
        night = cv.imread(TRAIN_DATA + "/" + location + "/" + times[1] + "/" + night, 0)
        print(location, day.shape[0], day.shape[1], night.shape[0], night.shape[1])



for location in test_dirs:
    times = os.listdir(TEST_DATA + "/" + location)
    days = os.listdir(TEST_DATA + "/" + location + "/" + times[0])
    nights = os.listdir(TEST_DATA + "/" + location + "/" + times[1])
    #lengths.append([len(days), len(nights)])
    #print("Test", location, len(days), len(nights))
    for day, night in zip(days, nights):
        day = cv.imread(TEST_DATA + "/" + location + "/" + times[0] + "/" + day, 0)
        night = cv.imread(TEST_DATA + "/" + location + "/" + times[1] + "/" + night, 0)
        print(location, day.shape[0], day.shape[1], night.shape[0], night.shape[1])


for location in valid_dirs:
    times = os.listdir(VALID_DATA + "/" + location)
    days = os.listdir(VALID_DATA + "/" + location + "/" + times[0])
    nights = os.listdir(VALID_DATA + "/" + location + "/" + times[1])
    #lengths.append([len(days), len(nights)])
    #print("Valid", location, len(days), len(nights))
    for day, night in zip(days, nights):
        day = cv.imread(VALID_DATA + "/" + location + "/" + times[0] + "/" + day, 0)
        night = cv.imread(VALID_DATA + "/" + location + "/" + times[1] + "/" + night, 0)
        print(location, day.shape[0], day.shape[1], night.shape[0], night.shape[1])

'''

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
                imgs = choices(night_imgs, k=6)
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
                #imgs = choices(night_imgs, k=5)
                for nightimg in night_imgs:
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
                imgs = choices(night_imgs, k=25)
                for nightimg in imgs:
                    daypath = VALID_DATA + "/" + locations + "/" + day + "/" + dayimg
                    nightpath = VALID_DATA + "/" + local + "/" + night + "/" + nightimg
                    yield daypath, nightpath

'''counter = 0
for day, night in tr_get_next_positive_path():
    counter += 1
print(counter)

counter = 0
for day, night in tr_get_next_negative_path():
    counter += 1
print(counter)

counter = 0
for day, night in va_get_next_positive_path():
    counter += 1
print(counter)

counter = 0
for day, night in va_get_next_negative_path():
    counter += 1
print(counter)

counter = 0
for day, night in te_get_next_positive_path():
    counter += 1
print(counter)

counter = 0
for day, night in te_get_next_negative_path():
    counter += 1
print(counter)'''


day = cv.imread("../data/train/00000850/Day/20151101_142506.jpg", 0)
night = cv.imread("../data/train/00000850/Night/20151101_072507.jpg", 0)


night_e = cv.equalizeHist(night)
cv.imwrite("testImgs/night_e.png", night_e)


clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
night_c = clahe.apply(night)
cv.imwrite("testImgs/night_clahe.png", night_c)


day_fb = cv.fastNlMeansDenoising(day, 45.0,10,21)
night_fb = cv.fastNlMeansDenoising(night_c, 45.0,10,21)
cv.imwrite("testImgs/day_fb.png", day_fb)
cv.imwrite("testImgs/night_fb.png", night_fb)


day_mb = cv.medianBlur(day, 9)
night_mb = cv.medianBlur(night_c, 9)
cv.imwrite("testImgs/day_mb.png", day_mb)
cv.imwrite("testImgs/night_mb.png", night_mb)


day_gb = cv.GaussianBlur(day,(5,5),0)
night_gb = cv.GaussianBlur(night_c,(5,5),0)
cv.imwrite("testImgs/day_gb.png", day_gb)
cv.imwrite("testImgs/night_gb.png", night_gb)