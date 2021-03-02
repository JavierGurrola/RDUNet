import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, Sampler
from skimage.util import view_as_windows
from utils import load_image
from transforms import ToTensor


def data_augmentation(image):
    augmented_images_arrays, augmented_images_list = [], []
    to_transform = [image, np.rot90(image, axes=(1, 2))]

    for t in to_transform:
        t_ud = t[:, ::-1, ...]
        t_lr = t[:, :, ::-1, ...]
        t_udlr = t_ud[:, :, ::-1, ...]

        flips = [t_ud, t_lr, t_udlr]
        augmented_images_arrays.extend(flips)

    augmented_images_arrays.extend(to_transform)

    for img in augmented_images_arrays:
        img_unbatch = list(img)
        augmented_images_list.extend(img_unbatch)

    return augmented_images_list


def create_patches(image, patch_size, step):
    image = view_as_windows(image, patch_size, step)
    h, w = image.shape[:2]
    image = np.reshape(image, (h * w, patch_size[0], patch_size[1], patch_size[2]))

    return image


class DataSampler(Sampler):

    def __init__(self, data_source, num_samples=None):
        super().__init__(data_source)
        self.data_source = data_source
        self._num_samples = num_samples
        self.rand = np.random.RandomState(0)
        self.perm = []

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self._num_samples is not None:
            while len(self.perm) < self._num_samples:
                perm = self.rand.permutation(n).astype('int32').tolist()
                self.perm.extend(perm)
            idx = self.perm[:self._num_samples]
            self.perm = self.perm[self._num_samples:]
        else:
            idx = self.rand.permutation(n).astype('int32').tolist()

        return iter(idx)

    def __len__(self):
        return self.num_samples


class NoisyImagesDataset(Dataset):
    def __init__(self, files, channels, patch_size, transform=None, noise_transform=None):
        self.channels = channels
        self.patch_size = patch_size
        self.transform = transform
        self.noise_transforms = noise_transform
        self.to_tensor = ToTensor()
        self.dataset = {'image': [], 'noisy': []}
        self.load_dataset(files)

    def __len__(self):
        return len(self.dataset['image'])

    def __getitem__(self, idx):
        image, noisy = self.dataset.get('image')[idx], self.dataset.get('noisy')[idx]
        sample = {'image': image, 'noisy': noisy}
        if self.transform is not None:
            sample = self.transform(sample)
        sample = self.to_tensor(sample)

        return sample.get('noisy'), sample.get('image')

    def load_dataset(self, files):
        patch_size = (self.patch_size, self.patch_size, self.channels)
        for file in tqdm(files):
            image = load_image(file, self.channels)
            if image is None:
                continue

            image = create_patches(image, patch_size, step=self.patch_size)
            sample = {'image': image, 'noisy': None}

            for noise_transform in self.noise_transforms:
                _sample = noise_transform(sample)
                image, noisy = _sample['image'], _sample['noisy']
                image, noisy = list(image), list(noisy)

                self.dataset['image'].extend(image)
                self.dataset['noisy'].extend(noisy)
