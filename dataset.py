import json
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

def get_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


class DatasetBuilder(Dataset):

    def __init__(self, data_path, image_size, logger):
        self.data = get_data(data_path)
        self.image_size = image_size
        self.logger = logger

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        try:
            image_left, image_right = self.read_images(item)
            image_left = self.preprocess_image(image_left)
            image_right = self.preprocess_image(image_right)
        except:
            self.logger.info(f"Can not read images: {self.data[item]}")
            raise
        return image_left, image_right

    def read_images(self, item):
        images_path = self.data[item]
        image_left = cv.imread(images_path[0])
        image_right = cv.imread(images_path[1])

        return image_left, image_right

    def preprocess_image(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (self.image_size[0], self.image_size[1]))
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.
        return image


def  create_dataloaders(data_path, image_size, batch_size, logger):

    logger.info("Reading data...")
    data = DatasetBuilder(data_path, image_size, logger)

    data_loader = DataLoader(dataset=data, pin_memory=True,
                              batch_size=batch_size, shuffle=True)

    return data_loader
