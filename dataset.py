import json
import cv2 as cv
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torch

from utils.utils import scale_image


def get_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image_left, image_right):
        if random.random() < self.p:
            image_right = np.fliplr(image_right)
            image_left = np.fliplr(image_left)
        return image_left, image_right


class RandomBrightness(object):
    def __init__(self, p=0.5, low_value=0.5, high_value=2.0):
        self.p = p
        self.low_value = low_value
        self.high_value = high_value

    def __call__(self, image_left, image_right):
        if random.random() < self.p:
            scale = np.random.uniform(low=self.low_value, high=self.high_value)
            image_left = np.uint8(np.float64(image_left) * scale)
            image_right = np.uint8(np.float64(image_right) * scale)
        return image_left, image_right


class RandomGamma(object):
    def __init__(self, p=0.5, low_value=0.5, high_value=2.0):
        self.p = p
        self.low_value = low_value
        self.high_value = high_value

    def __call__(self, image_left, image_right):
        if random.random() < self.p:
            gamma = np.random.uniform(low=self.low_value, high=self.high_value)
            image_left = np.uint8(np.float64(image_left) ** gamma)
            image_right = np.uint8(np.float64(image_right) ** gamma)
        return image_left, image_right


class DatasetBuilder(Dataset):

    def __init__(self, data_path, image_size, augment, logger):
        self.data = get_data(data_path)
        self.image_size = image_size
        self.logger = logger
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        try:
            image_left, image_right = self.read_images(item)
            if self.augment:
                image_left, image_right = self.augment_image(image_left, image_right)
            images_left = self.preprocess_image(image_left)
            images_right = self.preprocess_image(image_right)
        except:
            self.logger.info(f"Can not read images: {self.data[item]}")
            raise
        return images_left, images_right

    def read_images(self, item):
        images_path = self.data[item]
        image_left = cv.imread(images_path[0])
        image_right = cv.imread(images_path[1])
        return image_left, image_right

    def augment_image(self, image_left, image_right):
        image_left, image_right = RandomHorizontalFlip(p=self.augment["HorizonalFlip"]["p"])(image_left, image_right)
        image_left, image_right = RandomBrightness(p=self.augment["RandomBrightness"]["p"],
                                                   low_value=self.augment["RandomBrightness"]["low_value"],
                                                   high_value=self.augment["RandomBrightness"]["high_value"])\
                                                   (image_left, image_right)
        image_left, image_right = RandomGamma(p=self.augment["RandomGamma"]["p"],
                                              low_value=self.augment["RandomGamma"]["low_value"],
                                              high_value=self.augment["RandomGamma"]["high_value"]) \
                                              (image_left, image_right)
        return image_left, image_right

    def preprocess_image(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (self.image_size[0], self.image_size[1]))
        images_scaled = scale_image(image)
        images_normalized = [img_scaled / 255. for img_scaled in images_scaled]
        images_normalized = [np.transpose(img_norm, (2, 0, 1)) for img_norm in images_normalized]
        return images_normalized


def create_dataloaders(data_path, image_size, batch_size, augment, logger):

    logger.info("Reading data...")
    data = DatasetBuilder(data_path, image_size, augment, logger)

    data_loader = DataLoader(dataset=data, pin_memory=True,
                             batch_size=batch_size, shuffle=True)

    return data_loader
