import torch
import glob
import pandas as pd
import os
import numpy as np
import random

path_to_images = (
    ""
)
path_to_annotations = ""

training_data = np.zeros((len(os.listdir(path_to_images)), 2), dtype=object)
image_list = os.listdir(path_to_images)
random.shuffle(image_list)
for idx, i in enumerate(image_list):
    training_data[idx, 0] = os.path.join(path_to_images, i)
    training_data[idx, 1] = os.path.join(path_to_annotations, i)
validation_data = training_data[int(len(training_data) * 0.95) :]
training_data = training_data[: int(len(training_data) * 0.95)]

pd.DataFrame(training_data).to_csv(
    "train.csv",
    index=False,
)

pd.DataFrame(validation_data).to_csv(
    "valid.csv",
    index=False,
)
