import random
import torch
import numpy as np


class AdditiveWhiteGaussianNoise(object):
    """Additive white gaussian noise generator."""
    def __init__(self, noise_level, fix_sigma=False, clip=False):
        self.noise_level = noise_level
        self.fix_sigma = fix_sigma
        self.rand = np.random.RandomState(1)
        self.clip = clip
        if not fix_sigma:
            self.predefined_noise = [i for i in range(5, noise_level + 1, 5)]

    def __call__(self, sample):
        """
        Generates additive white gaussian noise in
        :param sample:
        :return:
        """
        image = sample.get('image')

        if image.ndim == 4:                 # if 'image' is a batch of images, we set a different noise level per image
            batches = image.shape[0]        # (Samples, Height, Width, Channels) or (Samples, Channels, Height, Width)
            if self.fix_sigma:
                sigma = self.noise_level * np.ones((batches, 1, 1, 1))
            else:
                sigma = np.random.choice(self.predefined_noise) * np.ones((batches, 1, 1, 1))
            noise = self.rand.normal(0., 1., size=image.shape)
            noise = noise * sigma
        else:                               # else, 'image' is a simple image
            if self.fix_sigma:              # (Height, Width, Channels) or (Channels , Height, Width)
                sigma = self.noise_level
            else:
                sigma = self.rand.randint(5, self.noise_level)
            noise = self.rand.normal(0., sigma, size=image.shape)

        noisy = image + noise

        if self.clip:
            noisy = np.clip(noisy, 0., 255.)

        return {'image': image, 'noisy': noisy.astype('float32')}


class RandomCrop(object):
    """Random crop generator."""

    def __init__(self, crop_size):
        assert isinstance(crop_size, (list, tuple)) or isinstance(crop_size, int)
        if isinstance(crop_size, (list, tuple)):
            self.crop_size = crop_size
        elif isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)

    def __call__(self, sample):
        """
        Generates a random patch from 'image' with size 'crop_size'.
        :param sample: image to crop and its respective image map.
        :return: sample with cropped image and edge map.
        """
        image = sample.get('image')
        height, width = image.shape[:2]
        new_height, new_width = self.crop_size

        top = np.random.randint(0, height - new_height)
        left = np.random.randint(0, width - new_width)

        image = image[top: top + new_height, left: left + new_width]

        return {'image': image, 'noisy': sample.get('noisy')}


class ToTensor(object):
    """Convert data sample to pytorch tensor"""
    def __call__(self, sample):
        image, noisy = sample.get('image'), sample.get('noisy')
        image = torch.from_numpy(image.transpose((2, 0, 1)).astype('float32') / 255.)

        if noisy is not None:
            noisy = torch.from_numpy(noisy.transpose((2, 0, 1)).astype('float32') / 255.)

        return {'image': image, 'noisy': noisy}


class RandomVerticalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.uniform(0., 1.) < self.p:
            image, noisy = sample.get('image'), sample.get('noisy')
            image = np.flipud(image)

            if noisy is not None:
                noisy = np.flipud(noisy)

            return {'image': image, 'noisy': noisy}

        return sample


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.uniform(0., 1.) < self.p:
            image, noisy = sample.get('image'), sample.get('noisy')
            image = np.fliplr(image)

            if noisy is not None:
                noisy = np.fliplr(noisy)

            return {'image': image, 'noisy': noisy}

        return sample


class RandomRot90(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.uniform(0., 1.) < self.p:
            image, noisy = sample.get('image'), sample.get('noisy')
            image = np.rot90(image)

            if noisy is not None:
                noisy = np.rot90(noisy)

            return {'image': image, 'noisy': noisy}

        return sample
