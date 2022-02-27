import torch
from dataset import BasicDataset
from random import choices
import os

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
                #imgs = choices(night_imgs, k=40)
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

import telegram_send

if __name__ == "__main__":
    tr_pos_gen = tr_get_next_positive_path
    tr_neg_gen = tr_get_next_negative_path

    va_pos_gen = va_get_next_positive_path
    va_neg_gen = va_get_next_negative_path

    te_pos_gen = te_get_next_positive_path
    te_neg_gen = te_get_next_negative_path

    counter = 0
    for day, night in tr_pos_gen():
        print(f"tr_pos_{counter}", end='\r')
        counter +=1
    
    print("tr pos final: ", counter)

    counter = 0
    for day, night in tr_neg_gen():
        print(f"tr_neg_{counter}", end='\r')
        counter +=1
    
    print("tr neg final: ", counter)

    counter = 0
    for day, night in va_pos_gen():
        print(f"va_pos_{counter}", end='\r')
        counter +=1
    
    print("va pos final: ", counter)

    counter = 0
    for day, night in va_neg_gen():
        print(f"va_neg_{counter}", end='\r')
        counter +=1
    
    print("va neg final: ", counter)

    counter = 0
    for day, night in te_pos_gen():
        print(f"te_pos_{counter}", end='\r')
        counter +=1
    
    print("te pos final: ", counter)

    counter = 0
    for day, night in te_neg_gen():
        print(f"te_neg_{counter}", end='\r')
        counter +=1

    print("te neg final: ", counter)

    print("making tr_data")
    tr_dataset = BasicDataset(train="Train")
    print("saving tr_data")
    torch.save(tr_dataset, "../data/datasets/tr_unprocessed_bal_144x256.pt")
    print("tr_data saved")
    telegram_send.send(messages=[f"Process Completed ... saved tr_data"])

    print("making va_data")
    va_dataset = BasicDataset(train="Valid")
    print("saving va_data")
    torch.save(va_dataset, "../data/datasets/va_unprocessed_bal_bot3_144x256.pt")
    print("va_data saved")
    telegram_send.send(messages=[f"Process Completed ...  saved va_data"])
    
    print("making tr_data")
    tr_dataset = BasicDataset(train="Train")
    print("saving tr_data")
    torch.save(tr_dataset, "../data/datasets/tr_unprocessed_bal_144x256.pt")
    print("tr_data saved")
    telegram_send.send(messages=[f"Process Completed ... saved te_data"])