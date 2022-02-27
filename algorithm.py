from skimage.metrics import structural_similarity
import imutils
import cv2, os
from random import choices
import wandb

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

def pipeline(dayPath, nightPath):
    day = cv2.imread(dayPath)
    night = cv2.imread(nightPath)
    h = 144
    w = 256
    re = (w,h)
    day = cv2.resize(day ,re, interpolation = cv2.INTER_AREA)
    night = cv2.resize(night ,re, interpolation = cv2.INTER_AREA)
    day_g = cv2.cvtColor(day, cv2.COLOR_BGR2GRAY)
    night_g = cv2.cvtColor(night, cv2.COLOR_BGR2GRAY)
    return day_g, night_g


def similarity(day, night, type):
    (score, diff) = structural_similarity(day, night, gaussian_weights=True, sigma=0.1, use_sample_covariance=False, data_range=255,full=True)
    print("SSIM: {}".format(score))

    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print("got contours")

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(day, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(night, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    rec = {f"{type} Reference": wandb.Image(day), f"{type} Test":wandb.Image(night), f"{type} Thresh": wandb.Image(thresh), f"{type} Diff": wandb.Image(diff), f"{type} Score": score}
    return rec
    


if __name__ == "__main__":
    gens = [tr_get_next_positive_path, tr_get_next_negative_path, va_get_next_positive_path, va_get_next_negative_path]
    types = ['train_pos', 'train_neg', 'val_pos', 'val_neg']

    for gen, type in zip(gens, types):
        print(type)
        for day, night in gen():
            day_g, night_g = pipeline(day, night)
            dict = similarity(day_g, night_g, type)







'''day = cv2.imread('../data/train/00000850/Day/20151101_142506.jpg')
night = cv2.imread('../data/train/00001323/Night/20151101_221025.jpg') #/data/train/00000850/Night/20151101_072507.jpg /data/train/00001323/Night/20151101_221025.jpg
h = 144
w = 256
print("read images")
re = (w,h)
day = cv2.resize(day ,re, interpolation = cv2.INTER_AREA)
night = cv2.resize(night ,re, interpolation = cv2.INTER_AREA)

# convert the images to grayscale
day_g = cv2.cvtColor(day, cv2.COLOR_BGR2GRAY)
night_g = cv2.cvtColor(night, cv2.COLOR_BGR2GRAY)
print("fixed images")
#(score, diff) = compare_ssim(day_g, night_g, full=True)
print("compared images")
(score,diff)=structural_similarity(day_g, night_g, gaussian_weights=True, sigma=0.1, use_sample_covariance=False, data_range=255,full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))


thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("got contours")

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(day, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(night, (x, y), (x + w, y + h), (0, 0, 255), 2)
# show the output images
print('writing images')

cv2.imwrite("testImgs/day1.png", day)
cv2.imwrite("testImgs/night1.png", night)
cv2.imwrite("testImgs/diff1.png", diff)
cv2.imwrite("testImgs/thresh1.png", thresh)
cv2.waitKey(0)'''