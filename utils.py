import random
import torch
import numpy as np
from skimage import io, color, img_as_ubyte


def load_image(image_path, channels):
    """
    Load image and change it color space from RGB to Grayscale if necessary.
    :param image_path: str
        Path of the image.
    :param channels: int
        Number of channels (3 for RGB, 1 for Grayscale)
    :return: numpy array
        Image loaded.
    """
    image = io.imread(image_path)

    if image.ndim == 3 and channels == 1:       # Convert from RGB to Grayscale and expand dims.
        image = img_as_ubyte(color.rgb2gray(image))
        return np.expand_dims(image, axis=-1)
    elif image.ndim == 2 and channels == 1:     # Handling grayscale images if needed.
        if image.dtype != 'uint8':
            image = img_as_ubyte(image)
        return np.expand_dims(image, axis=-1)

    return image


def mod_crop(image, mod):
    """
    Crops image according to mod to restore spatial dimensions
    adequately in the decoding sections of the model.
    :param image: numpy array
        Image to crop.
    :param mod: int
        Module for padding allowed by the number of
        encoding/decoding sections in the model.
    :return: numpy array
        Copped image
    """
    size = image.shape[:2]
    size = size - np.mod(size, mod)
    image = image[:size[0], :size[1], ...]

    return image


def mod_pad(image, mod):
    """
    Pads image according to mod to restore spatial dimensions
    adequately in the decoding sections of the model.
    :param image: numpy array
        Image to pad.
    :param mod: int
        Module for padding allowed by the number of
        encoding/decoding sections in the model.
    :return: numpy  array, tuple
        Padded image, original image size.
    """
    size = image.shape[:2]
    h, w = np.mod(size, mod)
    h, w = mod - h, mod - w
    if h != mod or w != mod:
        if image.ndim == 3:
            image = np.pad(image, ((0, h), (0, w), (0, 0)), mode='reflect')
        else:
            image = np.pad(image, ((0, h), (0, w)), mode='reflect')

    return image, size


def set_seed(seed=1):
    """
    Sets all random seeds.
    :param seed: int
        Seed value.
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_ensemble(image, normalize=True):
    """
    Create image ensemble to estimate denoised image.
    :param image: numpy array
        Noisy image.
    :param normalize: bool
        Normalize image to range [0., 1.].
    :return: list
        Ensemble of noisy image transformed.
    """
    img_rot = np.rot90(image)
    ensemble_list = [
        image, np.fliplr(image), np.flipud(image), np.flipud(np.fliplr(image)),
        img_rot, np.fliplr(img_rot), np.flipud(img_rot), np.flipud(np.fliplr(img_rot))
    ]

    ensemble_transformed = []
    for img in ensemble_list:
        if img.ndim == 2:                                           # Expand dims for channel dimension in gray scale.
            img = np.expand_dims(img.copy(), 0)                     # Use copy to avoid problems with reverse indexing.
        else:
            img = np.transpose(img.copy(), (2, 0, 1))               # Channels-first transposition.
        if normalize:
            img = img / 255.

        img_t = torch.from_numpy(np.expand_dims(img, 0)).float()    # Expand dims again to create batch dimension.
        ensemble_transformed.append(img_t)

    return ensemble_transformed


def separate_ensemble(ensemble, return_single=False):
    """
    Apply inverse transforms to predicted image ensemble and average them.
    :param ensemble: list
        Predicted images, ensemble[0] is the original image,
        and ensemble[i] is a transformed version of ensemble[i].
    :param return_single: bool
        Return also ensemble[0] to evaluate single prediction
    :return: numpy array or tuple of numpy arrays
        Average of the predicted images, original image denoised.
    """
    ensemble_np = []

    for img in ensemble:
        img = img.squeeze()                     # Remove additional dimensions.
        if img.ndim == 3:                       # Transpose if necessary.
            img = np.transpose(img, (1, 2, 0))

        ensemble_np.append(img)

    # Apply inverse transforms to vertical and horizontal flips.
    img = ensemble_np[0] + np.fliplr(ensemble_np[1]) + np.flipud(ensemble_np[2]) + np.fliplr(np.flipud(ensemble_np[3]))

    # Apply inverse transforms to 90º rotation, vertical and horizontal flips
    img = img + np.rot90(ensemble_np[4], k=3) + np.rot90(np.fliplr(ensemble_np[5]), k=3)
    img = img + np.rot90(np.flipud(ensemble_np[6]), k=3) + np.rot90(np.fliplr(np.flipud(ensemble_np[7])), k=3)

    # Average and clip final predicted image.
    img = img / 8.
    img = np.clip(img, 0., 1.)

    if return_single:
        return img, ensemble_np[0]
    else:
        return img


def predict_ensemble(model, ensemble, device):
    """
    Predict batch of images from an ensemble.
    :param model: torch Module
        Trained model to estimate denoised images.
    :param ensemble: list
        Images to estimate.
    :param device: torch device
        Device of the trained model.
    :return: list
        Estimated images of type numpy ndarray.
    """
    y_hat_ensemble = []

    for x in ensemble:
        x = x.to(device)

        with torch.no_grad():
            y_hat = model(x)
            y_hat_ensemble.append(y_hat.cpu().detach().numpy().astype('float32'))

    return y_hat_ensemble
