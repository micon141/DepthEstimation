import torch
import cv2 as cv
import numpy as np
import random
import json

from torch.utils.data import Dataset, DataLoader


def get_data(data_path):
    """
    :param data_path: json file
    :return: data
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def preprocess_image(image, image_size):
    """
    :param image: Raw image that have to be preprocessed in order to run training or inference
    :param image_size: Input size of the network
    :return: images_normalized: Preprocessed image
    """
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (image_size[0], image_size[1]))
    images_normalized = image / 255.
    images_normalized = np.transpose(images_normalized, (2, 0, 1))
    return images_normalized


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        """
        :param p: Probability between 0 and 1
        """
        self.p = p

    def __call__(self, image_left, image_right):
        if random.random() < self.p:
            image_right = np.fliplr(image_right)
            image_left = np.fliplr(image_left)
        return image_left, image_right


class RandomBrightness(object):
    def __init__(self, p=0.5, low_value=0.5, high_value=2.0):
        """
        :param p: Probability between 0 and 1
        :param low_value: Minimum possible value
        :param high_value: Maximum possible value
        """
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
        """
        :param p: Probability between 0 and 1
        :param low_value: Minimum possible value
        :param high_value: Maximum possible value
        """
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
            images_left = preprocess_image(image_left, self.image_size)
            images_right = preprocess_image(image_right, self.image_size)
        except:
            self.logger.info(f"Can not read images: {self.data[item]}")
            raise
        return images_left, images_right

    def read_images(self, item):
        """
        :param item: Item to read
        :return: Read images
        """
        images_path = self.data[item]
        image_left = cv.imread(images_path[0])
        image_right = cv.imread(images_path[1])
        return image_left, image_right

    def augment_image(self, image_left, image_right):
        """
        :param image_left: Left image
        :param image_right: Right image
        :return: Augmented images
        """
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


def create_dataloaders(data_path, image_size, batch_size, augment, logger):
    """
    :param data_path: Path to json file
    :param image_size: Input size of the network
    :param batch_size: Batch size
    :param augment: Augmentation methods from config file
    :param logger: print logs
    :return:
    """

    logger.info("Reading data...")
    data = DatasetBuilder(data_path, image_size, augment, logger)

    data_loader = DataLoader(dataset=data, pin_memory=True,
                             batch_size=batch_size, shuffle=True)

    return data_loader
