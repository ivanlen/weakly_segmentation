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
    https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/

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
    def __init__(
            self,
            annotations_file,
            augment=None,
            x_transform=None,
            y_transform=None,
            split=None):

        self.data = pd.read_json(annotations_file)
        self.split = split
        if split:
            self.data = self.data[self.data['split'] == split]

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
        if self.augment:
            aug = self.augment(image=image, mask=label)
            image = aug['image']
            label = aug['mask']

        if self.x_transform:
            image = self.x_transform(Image.fromarray(image))
        if self.y_transform:
            label = self.y_transform(label)
        return image, label

