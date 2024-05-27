import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import glob
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
import re
from PIL import Image
class TinyImagenet(Dataset):
    def __init__(self, train_path, class_ls, transform_train=None, annotation=None):
        self.train_path = train_path
        self.transform_train = transform_train
        self.ls = class_ls
        self.annotation = annotation
        target_ls = []
        if self.annotation:
            f = open(self.annotation, 'r')
            lines = f.readlines()
            for line in lines:
                temp = line.split()
                name = temp[0]
                target_temp = temp[1]
                target = self.ls.index(target_temp)
                target_ls.append(target)

        self.target_ls = target_ls

    def __getitem__(self, index):
        train_img_path = self.train_path[index]
        if self.annotation:
            target = self.target_ls[index]

        else:
            target_temp = train_img_path.split('/')[-3]
            target = self.ls.index(target_temp)

        train_img = Image.open(train_img_path).convert("RGB")

        if self.transform_train:
            train_img = self.transform_train(train_img)

        return train_img, target

    def __len__(self):
        return len(self.train_path)