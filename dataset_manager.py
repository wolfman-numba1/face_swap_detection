from __future__ import division
import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as T


class CooccurenceDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir
        self.list_paths = []
        for f in os.listdir(self.root_dir + "faceswap/co"):
            co_and_fb = []
            if not f.startswith('.'):
                co_and_fb.append("faceswap/co/" + f)
                co_and_fb.append("faceswap/fb/" + f.replace(".npy", ".jpg"))
                self.list_paths.append(co_and_fb)
        for f in os.listdir(self.root_dir + "original/co"):
            co_and_fb = []
            if not f.startswith('.'):
                co_and_fb.append("original/co/" + f)
                co_and_fb.append("original/fb/" + f.replace(".npy", ".jpg"))
                self.list_paths.append(co_and_fb)

    def __len__(self):
        return len(self.list_paths)

    def __getitem__(self, idx):
        co_path = os.path.join(self.root_dir, self.list_paths[idx][0])
        fb_path = os.path.join(self.root_dir, self.list_paths[idx][1])

        ##cooccurence matrix
        np_image = np.load(co_path)
        np_image = np_image/255
        np_image = np.reshape(np_image, (6,256,256))
        np_image = torch.from_numpy(np_image)
        
        ##face border image
        fb_image = Image.open(fb_path)
        fb_image = self.transform(fb_image)


        if "faceswap" in co_path:
            label = 0
        elif "original" in co_path:
            label = 1

        return [np_image, fb_image], label
    
    transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)])

