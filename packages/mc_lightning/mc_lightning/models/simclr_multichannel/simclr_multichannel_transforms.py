import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torchvision import datasets, models, transforms
from collections import OrderedDict
import argparse
import os
import sys
import numpy as np

import random
import math
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image


def apply_lambda_iaa_transform(iaa_transform, unsqueeze=True):
    if unsqueeze:
        return transforms.Lambda(lambda x: iaa_transform(images=[x])[0])
    else:
        return transforms.Lambda(lambda x: iaa_transform(images=x))


class RandomResizedCropArray(object):
    """
    Modification of torchvision transforms to allow non-PIL input
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.shape[0] * img.shape[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            #             print('aspect_ratio: ', aspect_ratio)
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.shape[0] and h <= img.shape[1]:
                i = random.randint(0, img.shape[1] - h)
                j = random.randint(0, img.shape[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.shape[0] / img.shape[1]
        if (in_ratio < min(ratio)):
            w = img.shape[0]
            h = w / min(ratio)
        elif (in_ratio > max(ratio)):
            h = img.shape[1]
            w = h * max(ratio)
        else:  # whole image
            w = img.shape[0]
            h = img.shape[1]

        i = (img.shape[1] - h) // 2
        j = (img.shape[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        crop = img[i:i + h, j:j + w]
        resizer = iaa.size.Resize(self.size)
        resized_crop = resizer(images=np.expand_dims(crop, 0))[0]

        return resized_crop


class MultiplexIFNormalizerPerChannel(object):
    """
    Follow SHIFT's mIF pixel intensity normalization
    expects torch tensor input
    returns normalized tensor
    """

    def __init__(self, mean=0.25, stdev=0.125, in_channels=7):
        self.mean = mean
        self.stdev = stdev
        self.in_channels = in_channels

    def __call__(self, img):
        eps = 1e-5
        img_mean = img.view(self.in_channels, -1).mean(-1).view(self.in_channels, 1, 1)
        img_std = img.view(self.in_channels, -1).std(-1).view(self.in_channels, 1, 1) + eps

        normed_img = ((img - img_mean) * (self.stdev / img_std) + self.mean)
        clamped_img = torch.clamp(normed_img, 0, 1)

        return clamped_img


class SimCLRMultichannelTrainTransform(object):
    """
    """
    def np_to_torch(self, x):
        return torch.Tensor(x.astype(float)).permute(2,0,1)

    def torch_to_np(self, x):
        return x.permute(1,2,0).numpy()

    def __init__(self, crop_size=224, rhf_prob=0.5, et_alpha=10, et_sigma=1, shear=15):
        self.crop_size = crop_size
        self.et_alpha = et_alpha
        self.et_sigma = et_sigma
        self.rhf_prob = rhf_prob
        self.shear = shear

        et_alpha_sample = random.uniform(0, self.et_alpha)
        et_sigma_sample = random.uniform(0, self.et_sigma)

        affine_transform = iaa.Affine(rotate=(0, 360), shear=(-self.shear, self.shear))
        elastic_transformation = iaa.ElasticTransformation(alpha=(0, et_alpha_sample), sigma=(0, et_sigma_sample))
        flip = iaa.flip.Fliplr(p=self.rhf_prob)

        data_transforms = transforms.Compose([
            MultiplexIFNormalizerPerChannel(),
            self.torch_to_np,
            RandomResizedCropArray(size=self.crop_size),
            apply_lambda_iaa_transform(flip, unsqueeze=True),
            # apply_lambda_iaa_transform(affine_transform, unsqueeze=True),
            apply_lambda_iaa_transform(elastic_transformation, unsqueeze=True),
            self.np_to_torch,
        ])

        self.train_transform = data_transforms

    def __call__(self, sample):
        transform = self.train_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


class SimCLRMultichannelEvalTransform(object):
    """
    """
    def np_to_torch(self, x):
        return torch.Tensor(x.astype(float)).permute(2,0,1)

    def torch_to_np(self, x):
        return x.permute(1,2,0).numpy()

    def __init__(self, crop_size=224):
        self.crop_size = crop_size
        center_crop = iaa.size.CenterCropToFixedSize(self.crop_size, self.crop_size)

        data_transforms = transforms.Compose([
            MultiplexIFNormalizerPerChannel(),
            self.torch_to_np,
            apply_lambda_iaa_transform(center_crop, unsqueeze=True),
            self.np_to_torch,
        ])

        self.eval_transform = data_transforms

    def __call__(self, sample):
        transform = self.eval_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj
