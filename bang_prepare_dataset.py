"""Dataset for Bangla CLIP :: https://github.com/zabir-nabil/bangla-CLIP"""
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd

train_dataframe = {}
train_dataframe['caption'] = []
train_dataframe['image'] = []

valid_dataframe = {}
valid_dataframe['caption'] = []
valid_dataframe['image'] = []

# https://data.mendeley.com/datasets/rxxch9vw59/2
with open('/data/captions.json', encoding='utf-8') as fh:
    data = json.load(fh)

trn_split = int(0.8 * len(data))
for sample in tqdm(data[:trn_split]):
    fn = sample['filename']
    cp = sample['caption']
    my_file = Path(f"/data/images/{fn}")
    if my_file.is_file():
        for tc in cp:
            tc = tc.replace(',', ' ')
            train_dataframe['caption'].append(tc)
            train_dataframe['image'].append(f"/data/images/{fn}")

for sample in tqdm(data[trn_split:]):
    fn = sample['filename']
    cp = sample['caption']
    my_file = Path(f"/data/images/{fn}")
    if my_file.is_file():
        for vc in cp:
            vc = vc.replace(',', ' ')
            valid_dataframe['caption'].append(vc)
            valid_dataframe['image'].append(f"/data/images/{fn}")

# https://www.kaggle.com/datasets/almominfaruk/bnaturebengali-image-captioning-dataset?resource=download
# BNature dataset

lines = open("/data/caption/caption.txt", "r").readlines()

trn_split = int(0.8 * len(lines))
for line in tqdm(lines[:trn_split]):
    fn = [x.strip() for x in line.split()][0]
    cp = ' '.join([x.strip() for x in line.split()][1:])
    my_file = Path(f"/data/Pictures/{fn}")
    if my_file.is_file():
        cp = cp.replace(',', ' ')
        train_dataframe['caption'].append(cp)
        train_dataframe['image'].append(f"/data/Pictures/{fn}")

for line in tqdm(lines[trn_split:]):
    fn = [x.strip() for x in line.split()][0]
    cp = ' '.join([x.strip() for x in line.split()][1:])
    my_file = Path(f"/data/Pictures/{fn}")
    if my_file.is_file():
        cp = cp.replace(',', ' ')
        valid_dataframe['caption'].append(cp)
        valid_dataframe['image'].append(f"/data/Pictures/{fn}")


### flickr8k bang translation
ban_caps = pd.read_csv("./../BAN-Cap_captiondata.csv")
cap_ids = list(ban_caps['caption_id'])
ban_trans = list(ban_caps['bengali_caption'])
trn_split = int(0.8 * len(cap_ids))
for j in tqdm(range(len(cap_ids[:trn_split]))):
    ci = cap_ids[j].split("#")[0]
    bt = ban_trans[j]
    fn = ci
    cp = bt
    my_file = Path(f"/data/flickr8k_images/{fn}")
    if my_file.is_file():
        cp = cp.replace(',', ' ')
        train_dataframe['caption'].append(cp)
        train_dataframe['image'].append(f"/data/flickr8k_images/{fn}")

for j in tqdm(range(len(cap_ids[trn_split:]))):
    ci = cap_ids[j].split("#")[0]
    bt = ban_trans[j]
    fn = ci
    cp = bt
    my_file = Path(f"/data/flickr8k_images/{fn}")
    if my_file.is_file():
        cp = cp.replace(',', ' ')
        valid_dataframe['caption'].append(cp)
        valid_dataframe['image'].append(f"/data/flickr8k_images/{fn}")

        

train_dataframe = pd.DataFrame(train_dataframe)
valid_dataframe = pd.DataFrame(valid_dataframe)

print(train_dataframe.head())
print(valid_dataframe.head())

train_dataframe.to_csv('train_df_bang.csv', index = None)
valid_dataframe.to_csv('valid_df_bang.csv', index = None)