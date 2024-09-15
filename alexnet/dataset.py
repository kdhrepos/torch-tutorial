import os
import pandas as pd

import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision.io import read_image
from torch.utils.data import Dataset

training_data = datasets.ImageNet(
    root='data',
    split='train',
)

test_data = datasets.ImageNet(
    root='data',
    split='val',
)

# class ImageDataset(Dataset):
#     def __init__(self, img_dir: str, img_size: int, mode: str):
#         self.img_dir = img_dir
#         self.img_size = img_size
#         self.mode = mode

#         if self.mode == "train":
#             None
#         elif self.mode == "test":
#             None

#     def __len__(self):
#         return len(self.img_labels)
