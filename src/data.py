import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from torchvision.datasets.folder import default_loader
import json

class ImgSentDataset(Dataset):
    def __init__(self,
                 text_file,
                 feature_file=None,
                 shuffle_imgs=False,
                 random_imgs=False,
                 shot=-1):

        self.text_file = text_file
        self.feature_file = feature_file
        self.shuffle_imgs = shuffle_imgs
        self.random_imgs = random_imgs
        self.shot = shot
        self.raw_dataset = self.load_data()

    def load_data(self):
        data = []
        sentonly = True if self.feature_file is None else False

        # loading sentences
        with open(self.text_file, 'r') as f:
            sentences = [l.strip() for l in f.readlines()]

        N = len(sentences)

        # loading image features
        if not sentonly:
            #import pdb; pdb.set_trace()
            # with h5py.File(self.feature_file, "r") as f:
            #     imgs = torch.from_numpy(np.array(f['features']))

            # if self.shuffle_imgs:
            #     print('Ablation study: shuffling the imgs ')
            #     index = np.random.choice(N, N, replace=False)
            #     imgs = imgs[index]

            # if self.random_imgs:
            #     print('Ablation study: select random imgs ')
            #     index = np.random.choice(N, N, replace=True)
            #     imgs = imgs[index]

            # for sent, img in zip(sentences, imgs):
            #     d = {'sent': sent, 'img': img}
            #     data.append(d)
            
            with open(self.feature_file, "r") as outfile:
                clip_data = json.load(outfile)
                

            for k in clip_data:
                img = torch.tensor(clip_data[k]['image_feat'])
                for ic in range(len(clip_data[k]['captions'])):
                    sent = clip_data[k]['captions'][ic]
                    clip_feat = torch.tensor(clip_data[k]['lang_feat'][ic])

                    
                    d = {'sent': sent, 'img': img, 'clip_text_feat': clip_feat, 'img_key': k}
                    data.append(d)
            
            

        else:
            for sent in sentences:
                d = {'sent': sent}
                data.append(d)

        if self.shot > 0:
            index = np.random.choice(N, self.shot, replace=False)
            data = np.array(data)[index].tolist()

        return data


    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, item:int):
        datum = self.raw_dataset[item]

        return datum



