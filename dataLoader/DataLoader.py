import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class myDataset(Dataset):
    def __init__(self, data_dir, img_size,transform=None):
        self.sharp = os.path.join(data_dir, "sharp")
        self.blur = os.path.join(data_dir, "blur")
        self.transform = transform

        if len(os.listdir(self.sharp)) > 50000:
            self.sharp_image_list = os.listdir(self.sharp)[:50000]
            self.blur_image_list = os.listdir(self.blur)[:50000]
        else:
            self.sharp_image_list = os.listdir(self.sharp)
            self.blur_image_list = os.listdir(self.blur)


        self.data = []
        self.load_data()
        self.data = np.vstack(self.data).reshape(-1, 3, img_size, img_size)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def load_data(self):
        for i in range(len(self.sharp_image_list)):
            sharp_image_path = os.path.join(self.sharp, self.sharp_image_list[i])
            print(f"load image {i}, path is {sharp_image_path}")
            sharp_image = self.getImage(sharp_image_path)
            self.data.append(sharp_image)

    def __len__(self):
        return len(self.sharp_image_list)

    def __getitem__(self, idx):
        blur_image_path = os.path.join(self.blur, self.blur_image_list[idx])
        sharp_image_path = os.path.join(self.sharp, self.sharp_image_list[idx])
        blur_image = self.getImage(blur_image_path)
        sharp_image = self.getImage(sharp_image_path)
        return  sharp_image, blur_image
    def getImage(self, path):
        image = Image.open(path)

        if self.transform:
            image = self.transform(image)

        return image