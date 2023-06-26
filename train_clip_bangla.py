"""Training CLIP for Bangla :: https://github.com/zabir-nabil/bangla-CLIP"""
import torch
import torch.nn as nn
import torch.optim as optim 
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import time
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from dataset import CLIPDataset, get_transforms
import config as CFG
from CLIP_model import CLIPModel

tokenizer = AutoTokenizer.from_pretrained(CFG.text_encoder_model)
model = AutoModel.from_pretrained(CFG.text_encoder_model)


inputs = tokenizer("আমি বাংলায় গান গাই আমি আমার আমিকে চিরদিন এই বাংলায় খুঁজে পাই", return_tensors="pt") 

print(inputs)
print("Input shape ", inputs["input_ids"].shape)
print("Decoded text ", tokenizer.decode(inputs['input_ids'][0]))

outputs = model(**inputs)

text_encoder_n_dim = outputs.last_hidden_state
print("Last hidden state shape ", text_encoder_n_dim.shape)


def make_train_valid_dfs():
    train_dataframe = pd.read_csv('train_df_bang.csv', encoding='utf8').dropna()
    valid_dataframe = pd.read_csv('valid_df_bang.csv', encoding='utf8').dropna()

    print(train_dataframe.head())
    print(valid_dataframe.head())

    return train_dataframe, valid_dataframe

def custom_collate_fn(samples):
    img, caption = zip(*samples)

    token_list = tokenizer(caption, padding = True) 
    img = torch.stack(img)

    text = torch.Tensor(token_list["input_ids"]).long()
    mask = torch.Tensor(token_list["attention_mask"]).long()

    return img, text, mask

def build_data_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
        collate_fn=custom_collate_fn
    )
    return dataloader

def train_and_val_model(
    model, criterion, train_loader, val_loader, optimizer, num_epochs=10, scheduler=None
):
    since = time.time()
    best_loss = float('inf')

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
       
        model.train()

        running_loss = 0.0

        for sample in train_loader:
            input, texts, masks = sample
            batch_size = input.size(0)
            input = input.to(device)  
            texts = texts.to(device)
            masks = masks.to(device)


            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                image_vec, text_vec = model(
                    input, texts , masks
                ) 
                logits = torch.matmul(text_vec, image_vec.T)
                
                targets = torch.arange(logits.size(0)).long().to(device)

                texts_loss = criterion(logits, targets)
                images_loss = criterion(logits.T, targets)
                loss = (images_loss + texts_loss) / 2.0  

                loss.backward()
                optimizer.step()
                if scheduler != None:
                    scheduler.step()
            
                running_loss += loss.item()

        train_loss = running_loss

        model.eval() 

        running_loss = 0.0

        with torch.no_grad():
            for sample in val_loader:
                input, texts, masks = sample

                input = input.to(
                    device
                ) 
                texts = texts.to(device)
                masks = masks.to(device)

                image_vec, text_vec = model(
                    input, texts , masks
                )

                with torch.set_grad_enabled(False):
                    logits = torch.matmul(text_vec, image_vec.T)

                    targets = torch.arange(logits.size(0)).long().to(device)

                    texts_loss = criterion(logits, targets)
                    images_loss = criterion(logits.T, targets)
                    loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)

                    # statistics
                    running_loss += loss.item()

        val_loss = running_loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_path = f"saved_models/{CFG.model_tag}_best_ep-{epoch}_loss-{round(val_loss, 5)}.pt"
            torch.save(model.state_dict(), best_model_path)
            print("Saved Best Model!")
            log.write("Model saved!\n")


        print(f"Epoch {epoch} :: Train Loss :: {train_loss} :: Validation Loss :: {val_loss}\n")
        log.write(f"Epoch {epoch} :: Train Loss :: {train_loss} :: Validation Loss :: {val_loss}\n")
        

        pbar.set_description(
            "train loss {:.4} val loss {:.4}".format(train_loss, val_loss)
        )
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    
    return model

train_df, valid_df = make_train_valid_dfs()
log = open(f"{CFG.log_tag}_log.txt", "a+")

print("building tokenizer")
tokenizer = AutoTokenizer.from_pretrained(CFG.text_tokenizer)

print("building train loader")
train_loader = build_data_loaders(train_df, tokenizer, mode="train")
print("building valid loader")
valid_loader = build_data_loaders(valid_df, tokenizer, mode="valid")

print("building CLIP model, move to GPU.")
print(f"{CFG.model_tag} is training.")

inputs, text , mask = next(iter(train_loader))

print("Sample input shape ", inputs.shape)
print("Sample text shape", text.shape)

device = CFG.device


model = CLIPModel().to(device)

criterion = nn.CrossEntropyLoss() 

optimizer = optim.Adam(model.parameters(), lr=1e-4)

now = datetime.now()

num_epochs = 15


model = train_and_val_model(
    model, criterion, train_loader, valid_loader, optimizer, num_epochs=num_epochs, scheduler=None
)