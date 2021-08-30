import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.labelling import label_color_map

seed = 18
np.random.seed(seed)


def augment_preprocess_generator():
    """
    input of transform must be ndarray
    :return:
    """
    transform = A.Compose([
        A.RandomRotate90(p=0.7),
        A.HorizontalFlip(p=0.5),
    ])
    return transform


def image_preprocess_transforms_generator(model_size=224):
    """
    checkout https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/ for normalization parameters

    :param model_size: at least 224
    :return:
    """
    assert model_size >= 224, f'image size must be >= 224, current{model_size}'
    preprocess = transforms.Compose([
        transforms.Resize((model_size, model_size)),  # input of resize has to be a Pillow Image
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess


def labels_preprocess_transforms_generator(model_size=224, mapper=label_color_map):
    """
    https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/


    :param model_size: at least 224
    :param mapper: {color: class} dict mapper
    :return:
    """
    assert model_size >= 224, f'image size must be >= 224, current{model_size}'

    def rgb_to_classes(x):
        w = x.shape[1]
        h = x.shape[0]
        target = x
        mapping = {tuple(c): t for c, t in zip(list(mapper.values()), range(len(mapper)))}

        mask = np.ones((h, w)).astype(int) * 5  # 5 is background
        for k in mapping:
            color_filter = np.all(target == k, axis=2)
            mask[color_filter] = mapping[k]
        return mask

    # we use this transform because PIL input must be uint8 (0..255 range) type and here we are using int
    def cv2_resize_nearest(x):
        return cv2.resize(x, (model_size, model_size), interpolation=cv2.INTER_NEAREST)

    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: rgb_to_classes(x)),
        transforms.Lambda(lambda x: cv2_resize_nearest(x)),
        # transforms.Resize((model_size, model_size), Image.NEAREST),
        transforms.ToTensor(),
    ])
    return preprocess


class VaihingenDataset(Dataset):
    """
    Dataset class that reads the processed json file and it parsed from the given parameters.
    """

    def __init__(
            self,
            annotations_file,
            augment=None,
            x_transform=None,
            y_transform=None,
            split=None):
        """

        :param annotations_file: json processed annotations file
        :param augment: augmentation pipeline from albumentations
        :param x_transform: transformations of images
        :param y_transform: transformations of labels
        :param split: split to select from processed annotations.
        """

        self.data = pd.read_json(annotations_file)
        self.split = split
        if split:
            if split in ['t1', 't2', 'val']:
                self.data = self.data[self.data['split'] == split]
            elif split == 't1t2':
                d1 = self.data[self.data['split'] == 't1']
                d2 = self.data[self.data['split'] == 't2']
                self.data = pd.concat([d1, d2])

        # initial resampling to shuffle different splits
        self.data = self.data.sample(frac=1, random_state=seed)
        self.augment = augment
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.n_samples = len(self.data)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        img_path = sample.tile_im_path
        lab_path = sample.tile_lab_path
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(cv2.imread(lab_path), cv2.COLOR_BGR2RGB)
        oh_classes = torch.FloatTensor(sample.oh_classes)
        is_weak = torch.BoolTensor([sample.is_weak])
        if self.augment:
            aug = self.augment(image=image, mask=label)
            image = aug['image']
            label = aug['mask']

        if self.x_transform:
            image = self.x_transform(Image.fromarray(image))
        if self.y_transform:
            label = self.y_transform(label)
        return image, (label, oh_classes, is_weak)
