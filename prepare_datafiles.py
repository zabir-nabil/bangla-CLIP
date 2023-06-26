import json
from glob import glob
from tqdm import tqdm
import config as CFG
import os
import pandas as pd

train_folders = ["train_text_img_pairs_0_compressed", "train_text_img_pairs_1_compressed", "train_text_img_pairs_2_compressed", 
                    "train_text_img_pairs_3_compressed", "train_text_img_pairs_4_compressed",  "train_text_img_pairs_5_compressed", 
                    "train_text_img_pairs_6_compressed", "train_text_img_pairs_7_compressed", "train_text_img_pairs_8_compressed"]
caption_labels = {}
train_json = json.load(open(CFG.train_json))

for tj in train_json:
    cap = tj['caption']
    pro = tj['product']
    caption_labels[pro] = cap

train_dataframe = {}
train_dataframe['caption'] = []
train_dataframe['image'] = []
print("Processing training images.")
for tf in train_folders:
    print(tf)
    images_tf = list(glob(f"{CFG.dataset_root}/{tf}/*"))
    print("loading done")
    for img_p in tqdm(images_tf):
        cap = caption_labels[os.path.basename(img_p)]

        train_dataframe['caption'].append(cap)
        train_dataframe['image'].append(img_p)



val_folders = ["val_imgs"]
caption_labels = {}
val_json = json.load(open(CFG.val_json))

for tj in val_json:
    cap = tj['caption']
    pro = tj['product']
    caption_labels[pro] = cap
valid_dataframe = {}

valid_dataframe['caption'] = []
valid_dataframe['image'] = []
for tf in val_folders:
    print(tf)
    images_tf = list(glob(f"{CFG.dataset_root}/{tf}/*"))
    print("loading done")
    for img_p in tqdm(images_tf):
        cap = caption_labels[os.path.basename(img_p)]

        valid_dataframe['caption'].append(cap)
        valid_dataframe['image'].append(img_p)


print(valid_dataframe)

train_dataframe = pd.DataFrame(train_dataframe)
valid_dataframe = pd.DataFrame(valid_dataframe)

print(train_dataframe.head())
print(valid_dataframe.head())

train_dataframe.to_csv('train_df.csv', index = None)
valid_dataframe.to_csv('valid_df.csv', index = None)