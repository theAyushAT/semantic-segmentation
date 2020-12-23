import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt

class classDataset(Dataset):
    def __init__(self,csvpath,mode,height,width, mean_std,debug=False):
        """      
        Format of csv file:
        It contains 2 columns
        1. Path to image
        2. Path to mask

        """

        self.csv_file = (
            pd.read_csv(os.path.join(csvpath, mode + ".csv"))
            .iloc[:, :]
            .values
        )
        self.mean_std = mean_std
        self.height = height
        self.width = width
        self.mode = mode
        self.debug = debug


    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        
    def __len__(self):
        """returns length of CSV file"""
        return len(self.csv_file)

    def __getitem__(self, idx):

        image = cv2.imread(self.csv_file[idx, 0], cv2.IMREAD_COLOR) 
        # print(self.csv_file[idx, 1]) 
        label = (cv2.imread(
                self.csv_file[idx, 1], cv2.IMREAD_GRAYSCALE))//255
        # print(label.shape)
    
        if (
            image.shape[1] == self.width
            and image.shape[0] == self.height
            and label.shape[1] == self.width
            and label.shape[0] == self.height
        ):
            pass
        else:

            image = cv2.resize(
                image, (self.width, self.height))
            label = cv2.resize(
                label,
                (self.width, self.height),
                cv2.INTER_NEAREST,
            )
        
        if self.debug:
            self.show_sample(image, label)  
        transformation = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        
        image = transforms.Normalize(mean=self.mean_std[0], std=self.mean_std[1])(transformation(image))
        label = torch.from_numpy(label)
        
        if self.mode == 'train':
            # applying transforms
            augment = [
                transforms.RandomCrop(200),
                transforms.RandomHorizontalFlip(0.5)
            ]
            tfs = transforms.Compose(augment)
            # seed = random.randint(0, 2**32)
            # self._set_seed(seed)
            image = tfs(image)
            # self._set_seed(seed)
            label = tfs(label)
        
             

        sample = {
            "image": image,
            "label": label,
            "img_name": self.csv_file[idx, 0].split("/")[-1],
        }

        return sample

    def show_sample(self, image, label):
        plt.imshow(image)
        plt.show()
        plt.imshow(label)
        plt.show()




