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

# Todo: add support for spatial data augmentations


class classDataset(Dataset):
    def __init__(self,csvpath,mode,height,width,data_name, mean_std,debug=False):
        """

        Single format used, cfg file will contain directory of csv files        .
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
        self.label_mapping = torch.load(
            os.path.join("data", data_name + ".pth")
        )["label_map"]



    def __len__(self):
        """returns length of CSV file"""
        return len(self.csv_file)

    def __getitem__(self, idx):

        image = cv2.imread(self.csv_file[idx, 0], cv2.IMREAD_COLOR)  
        label = self.convert_label(cv2.imread(
                self.csv_file[idx, 1], cv2.IMREAD_GRAYSCALE))//255
    
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

        image = normalize(image,self.mean_std)
        label = torch.from_numpy(label)

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

    def convert_label(self, label):
        temp = label.copy()

        for k, v in self.label_mapping.items():
            label[temp == k] = v
        return label 

transformation = transforms.Compose(
    [transforms.ToPILImage(), transforms.ToTensor()])
    
def normalize(image,mean_std):
    return transforms.Normalize(mean=mean_std[0], std=mean_std[1])(transformation(image))