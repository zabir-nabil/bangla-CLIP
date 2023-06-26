"""PyTorch Dataset for CLIP Bangla :: https://github.com/zabir-nabil/bangla-image-search"""
from logging import config
import os
import cv2
import torch
import albumentations as A

import config as CFG
from normalizer import normalize # pip install git+https://github.com/csebuetnlp/normalizer


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        dataset for CLIP Bangla
        """
        self.image_filenames = image_filenames
        self.captions = [normalize(cap_sen) for cap_sen in list(captions)]

        self.transforms = transforms

    def __getitem__(self, idx):
        image = cv2.imread(f"{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        # item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        # item['caption'] = self.captions[idx]
        image = torch.tensor(image).permute(2, 0, 1).float()
        caption = self.captions[idx]

        return image, caption


    def __len__(self):
        return len(self.captions)



def get_transforms(mode="train"):
    if mode == "train":
        config = {
            'aug_prob' : 0.2
        }
        return A.Compose(
            [
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=config['aug_prob']),
                A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=config['aug_prob']),
                A.CoarseDropout(p=config['aug_prob']),
                A.GaussNoise(p=config['aug_prob']),
                A.ZoomBlur(p=config['aug_prob']),
                A.RandomFog(p=config['aug_prob']),
                A.Rotate((-20., 20.), p = 0.5),
                A.MotionBlur(p=config['aug_prob']),
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )


    