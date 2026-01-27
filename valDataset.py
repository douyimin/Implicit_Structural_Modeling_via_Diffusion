import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
import numpy as np


class valDataset(Dataset):
    def __init__(self, data_path):
        self.data_list2D = [os.path.join(data_path, fie_name) \
                            for fie_name in os.listdir(data_path)]

    def __len__(self):
        return len(self.data_list2D)

    def __getitem__(self, index):
        data = np.load(self.data_list2D[index])
        fault = data['fault'] * 2 - 1
        horiz = data['horiz'] * 2 - 1

        return horiz[None], fault[None]
