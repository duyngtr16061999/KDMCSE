import os
import torch
from PIL import Image
import numpy as np
from numpy import asarray
import clip

import pickle, gzip, json
from tqdm import tqdm

import json

# Load pre-trained CLIP
model_ = "ViT-L/14"


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load(model_, device=device, jit=False)

# with open("data/raw/annotations/captions_train2014.json",'r') as f:
#     train_anno = json.load(f)
# with open("data/raw/annotations/captions_val2014.json",'r') as f:
#     valid_anno = json.load(f)

# def from_anno_to_iddict(anno):
#     id_to_file_name = {}
    
#     for i in anno['images']:
#         id_to_file_name[i['id']] =i['file_name']
    
#     captions = []
    
#     for i in anno['annotations']:
#         captions.append([i['caption'],i['image_id'],id_to_file_name[i['image_id']]])    
        
#     id_to_caps = {}
    
#     for i in id_to_file_name:
#         id_to_caps[i] = {
#             "image":id_to_file_name[i],
#             "captions":[]
#         }
        
#     for i in captions:
#         id_to_caps[i[1]]["captions"].append(i[0])
        
#     return id_to_caps

# train_id_to_caps = from_anno_to_iddict(train_anno)
# valid_id_to_caps = from_anno_to_iddict(valid_anno)

# for k in tqdm(train_id_to_caps.keys()):
    
#     img = train_id_to_caps[k]['image']
#     captions = train_id_to_caps[k]['captions']
    
#     image_path = os.path.join("data/raw/train2014",img)
#     im = Image.open(image_path)
#     image = preprocess(im).unsqueeze(0).to(device)
#     image_features = clip_model.encode_image(image).squeeze(0).detach().cpu().numpy()
    
#     text = clip.tokenize(captions).to(device)
#     text_features = clip_model.encode_text(text).detach().cpu().numpy()
    
#     train_id_to_caps[k]['image_feat'] = image_features
#     train_id_to_caps[k]['lang_feat'] = text_features
    
# for k in tqdm(valid_id_to_caps.keys()):
    
#     img = valid_id_to_caps[k]['image']
#     captions = valid_id_to_caps[k]['captions']
    
#     image_path = os.path.join("data/raw/val2014",img)
#     im = Image.open(image_path)
#     image = preprocess(im).unsqueeze(0).to(device)
#     image_features = clip_model.encode_image(image).squeeze(0).detach().cpu().numpy()
    
#     text = clip.tokenize(captions).to(device)
#     text_features = clip_model.encode_text(text).detach().cpu().numpy()
    
#     valid_id_to_caps[k]['image_feat'] = image_features
#     valid_id_to_caps[k]['lang_feat'] = text_features
    
# for k in tqdm(train_id_to_caps.keys()):
    
#     train_id_to_caps[k]['image_feat'] = train_id_to_caps[k]['image_feat'].tolist()
#     train_id_to_caps[k]['lang_feat'] = train_id_to_caps[k]['lang_feat'].tolist()
    
# for k in tqdm(valid_id_to_caps.keys()):
    
#     valid_id_to_caps[k]['image_feat'] = valid_id_to_caps[k]['image_feat'].tolist()
#     valid_id_to_caps[k]['lang_feat'] = valid_id_to_caps[k]['lang_feat'].tolist()

model_name= model_.replace("/","").replace("-","_")

# with open(f'./data/train_coco_{model_name}.json', "w") as outfile:
#     json.dump(train_id_to_caps, outfile)

# with open(f'./data/val_coco_{model_name}.json', "w") as outfile:
#     json.dump(valid_id_to_caps, outfile)
    


import pandas as pd

data = pd.read_csv("data/captions.txt")  
images = data["image"].to_list()
captions = data["caption"].to_list()
id_to_caps = {}

for i in range(len(images)):
    id_to_caps[images[i][:-4]] = {
        "image":images[i],
        "captions":[]
    }
    
# for i in range(len(captions)):
#     id_to_caps[images[i][:-4]]["captions"].append(captions[i])
    
for i in range(len(captions)):
    if not type(captions[i]) == str:
        continue
    if len(captions[i]) > 300:
        continue
    id_to_caps[images[i][:-4]]["captions"].append(captions[i])
    


for k in tqdm(id_to_caps.keys()):
    img = id_to_caps[k]['image']
    captions = id_to_caps[k]['captions']

    image_path = os.path.join("data/raw/flickr30k",img)
    im = Image.open(image_path)
    image = preprocess(im).unsqueeze(0).to(device)
    image_features = clip_model.encode_image(image).squeeze(0).detach().cpu().numpy()

    text = clip.tokenize(captions).to(device)
    text_features = clip_model.encode_text(text).detach().cpu().numpy()

    id_to_caps[k]['image_feat'] = image_features
    id_to_caps[k]['lang_feat'] = text_features
        
for k in tqdm(id_to_caps.keys()):
    id_to_caps[k]['image_feat'] = id_to_caps[k]['image_feat'].tolist()
    id_to_caps[k]['lang_feat'] = id_to_caps[k]['lang_feat'].tolist()
    
with open(f'./data/flickr30k_{model_name}.json', "w") as outfile:
    json.dump(id_to_caps, outfile)