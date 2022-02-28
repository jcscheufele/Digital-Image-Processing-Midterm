from skimage.metrics import structural_similarity, normalized_mutual_information
import imutils
import cv2, os
from random import choices
import wandb
import numpy as np

from torch.nn import BCELoss

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

#clahe_d = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
def pipeline(dayPath, nightPath):
    day = cv2.imread(dayPath)
    night = cv2.imread(nightPath)
    h = 144
    w = 246
    re = (w,h)

    day = cv2.resize(day ,re, interpolation = cv2.INTER_AREA)
    night = cv2.resize(night ,re, interpolation = cv2.INTER_AREA)

    day_g = cv2.cvtColor(day, cv2.COLOR_BGR2GRAY)
    night_g = cv2.cvtColor(night, cv2.COLOR_BGR2GRAY)
    
    day_c = clahe.apply(day_g)
    night_c = clahe.apply(night_g)

    #day_e = cv2.equalizeHist(day_g)
    #night_e = cv2.equalizeHist(night_g)

    #day_b = cv2.medianBlur(day_c, 9)
    #night_b = cv2.medianBlur(night_c, 9)

    #cv2.fastNlMeansDenoising(day_g, day_g, 45.0,10,21)
    #cv2.fastNlMeansDenoising(night_e, night_e, 45.0, 10, 21)

    #day_g = cv2.resize(day_g, re, interpolation = cv2.INTER_AREA)
    #night_e = cv2.resize(night_e, re, interpolation = cv2.INTER_AREA)

    day_b = cv2.GaussianBlur(day_c,(5,5),0)
    night_b = cv2.GaussianBlur(night_c,(5,5),0)

    return day_b, night_b

'''def cosine_sim(day, night, type):
    score = np.dot(day, night)/

    rec = {f"{type} Reference": wandb.Image(day), f"{type} Test":wandb.Image(night), f"{type} Score": score} #, f"{type} Thresh": wandb.Image(thresh), f"{type} Diff": wandb.Image(diff)
    return rec'''


def similarity(day, night, type):
    #score = normalized_mutual_information(day, night)
    (score, diff) = structural_similarity(day, night, gaussian_weights=True, sigma=.1, use_sample_covariance=False, data_range=255, full=True)

    #diff = (diff * 255).astype("uint8")
    #thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    '''cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(day, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(night, (x, y), (x + w, y + h), (0, 0, 255), 2)'''
    
    rec = {f"{type} Reference": wandb.Image(day), f"{type} Test":wandb.Image(night), f"{type} Score": score} #, f"{type} Thresh": wandb.Image(thresh), f"{type} Diff": wandb.Image(diff)
    return rec
    

def sift(day, night, type):
    #keypoints_without_size = np.copy(day)
    sift = cv2.SIFT_create()
    key_day, desc_day = sift.detectAndCompute(day,None)
    #cv2.drawKeypoints(day,key_day, keypoints_without_size, color = (0, 255, 0))
    key_night, desc_night = sift.detectAndCompute(night,None)

    #feature matching
    bf = cv2.BFMatcher() #cv2.NORM_L1, crossCheck=True
    matches = bf.knnMatch(desc_day,desc_night, k=2)
    #matches = sorted(matches, key = lambda x:x.distance)
    good = []
    for sets in matches:
        sets = sorted(sets, key = lambda x:x.distance)
        if sets[0].distance < 0.75*sets[1].distance:
            good.append([sets[0]])
    if len(good) > 4:
        score = 1
    else:
        score = 0

    if type == "train_pos":
        truth = 1
    else:
        truth = 0

    if score == truth:
        corr = 1
    else:
        corr = 0

    print(f"Matches: {len(good)}, Day feats:{len(key_day)}, Night feats: {len(key_night)}, Score: {score}, Loss: {corr}")
    img3 = cv2.drawMatchesKnn(day, key_day, night, key_night, good, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #cv2.imwrite("testImgs/new.png",img3)
    rec = {f"{type} Combo": wandb.Image(img3), f"{type} Raw": len(good), f"{type} Score": score, "Correct": corr} #, f"{type} Thresh": wandb.Image(thresh), f"{type} Diff": wandb.Image(diff)
    return rec, corr



if __name__ == "__main__":
    wandb.init(project="CS545_Midterm", entity="jcscheufele")
    name = "Similarity Test Suite sift knn clahe + blur"
    wandb.run.name = name

    counter = 0
    num_corr = 0
    preds = 0
    for day, night in tr_get_next_positive_path():
        if (counter % 50 == 0): 
            day_g, night_g = pipeline(day, night)
            dict, corr = sift(day_g, night_g, 'train_pos')
            num_corr += corr
            preds += 1
            wandb.log(dict)
            print(f"step: {counter}         ", end='\r')
        counter += 1

    for day, night in tr_get_next_negative_path():
        if (counter % 50 == 0): 
            day_g, night_g = pipeline(day, night)
            dict, corr = sift(day_g, night_g, 'train_neg')
            num_corr += corr
            preds += 1
            wandb.log(dict)
            print(f"step: {counter}        ", end='\r')
        counter +=1
    
    acc = num_corr/preds
    print("Accuracy: ", acc)
    wandb.log({"Accuracy": acc})

    '''counter = 0
    for day, night in va_get_next_positive_path():
        if (counter % 15 == 0): 
            day_g, night_g = pipeline(day, night)
            dict = similarity(day_g, night_g, 'val_pos')
            wandb.log(dict)
            print(f"step: {counter}         ", end='\r')
        counter += 1

    counter = 0
    for day, night in va_get_next_negative_path():
        if (counter % 15 == 0): 
            day_g, night_g = pipeline(day, night)
            dict = similarity(day_g, night_g, 'val_neg')
            wandb.log(dict)
            print(f"step: {counter}       ", end='\r')
        counter += 1
'''


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