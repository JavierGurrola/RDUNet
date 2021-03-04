import os
import yaml
import torch
import numpy as np
import scipy.io as sio
from os.path import join
from model import RDUNet
from skimage import io
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from utils import build_ensemble, separate_ensemble, predict_ensemble, mod_pad, mod_crop


def predict(model, noisy_dataset, gt_dataset, device, padding, n_channels, results_path):
    # Load test datasets in format .mat
    X = sio.loadmat(noisy_dataset)['data'].flatten()
    Y = sio.loadmat(gt_dataset)['label'].flatten()

    y_pred, y_pred_ens = [], []
    psnr_list, ssim_list = [], []
    ens_psnr_list, ens_ssim_list = [], []

    n_images = len(X)
    multi_channel = True if n_channels == 3 else False

    for i in range(n_images):
        x, y = X[i], Y[i]

        if padding:
            x, size = mod_pad(x, 8)
        else:
            x, y = mod_crop(x, 8), mod_crop(y, 8)

        x = build_ensemble(x, normalize=False)

        with torch.no_grad():
            y_hat_ens = predict_ensemble(model, x, device)
            y_hat_ens, y_hat = separate_ensemble(y_hat_ens, return_single=True)

            if padding:
                y_hat = y_hat[:size[0], :size[1], ...]
                y_hat_ens = y_hat_ens[:size[0], :size[1], ...]

            y_pred.append(y_hat)
            y_pred_ens.append(y_hat_ens)
            psnr = peak_signal_noise_ratio(y, y_hat, data_range=1.)
            ssim = structural_similarity(y, y_hat, data_range=1., multichannel=multi_channel, gaussian_weights=True,
                                         sigma=1.5, use_sample_covariance=False)

            psnr_ens = peak_signal_noise_ratio(y, y_hat_ens, data_range=1.)
            ssim_ens = structural_similarity(y, y_hat_ens, data_range=1., multichannel=multi_channel,
                                             gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

            ens_psnr_list.append(psnr_ens)
            ens_ssim_list.append(ssim_ens)
            print('Image: {} - PSNR: {:.4f} - SSIM: {:.4f} - ens PSNR: {:.4f}'
                  ' - ens SSIM: {:.4f}'.format(i + 1, psnr, ssim, psnr_ens, ssim_ens))

    if results_path is not None:
        for i in range(n_images):
            y_hat = (255 * y_pred[i]).astype('uint8')
            y_hat_ens = (255 * y_pred_ens[i]).astype('uint8')

            y_hat = np.squeeze(y_hat)
            y_hat_ens = np.squeeze(y_hat_ens)

            os.makedirs(results_path, exist_ok=True)

            name = os.path.join(results_path, '{}_{:.4f}_{:.4f}.png'.format(i, psnr_list[i], ssim_list[i]))
            io.imsave(name, y_hat)

            name = os.path.join(results_path, '{}_{:.4f}_{:.4f}_ens.png'.format(i, ens_psnr_list[i], ens_ssim_list[i]))
            io.imsave(name, y_hat_ens)

    return np.mean(psnr_list), np.mean(ssim_list), np.mean(ens_psnr_list), np.mean(ens_ssim_list)


if __name__ == '__main__':
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    model_params = config['model']
    test_params = config['test']
    n_channels = model_params['channels']

    if n_channels == 3:
        model_path = join(test_params['pretrained models path'], 'model_color.pth')
        noisy_datasets = ['noisy_cbsd68_']  # Also tested in Kodak24 and Urban100 datasets.
        gt_datasets = ['cbsd68_label']

    else:
        model_path = join(test_params['pretrained models path'], 'model_gray.pth')
        noisy_datasets = ['noisy_set12_']   # Also tested in BSD68 and Kodak24 datasets.
        gt_datasets = ['set12_label']

    model_params = config['model']
    model = RDUNet(**model_params)

    device = torch.device(test_params['device'])
    print("Using device: {}".format(device))

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    for noisy_dataset, gt_dataset in zip(noisy_datasets, gt_datasets):
        print('Dataset: ', noisy_dataset)

        for noise_level in test_params['noise levels']:
            extension = '_color' if model_params['channels'] == 3 else '_gray'

            noisy_path = join(test_params['dataset path'], ''.join([noisy_dataset, str(noise_level), extension, '.mat']))
            label_path = join(test_params['dataset path'], ''.join([gt_dataset, extension, '.mat']))

            if test_params['save images']:
                save_path = join(
                    test_params['results path'], ''.join([noisy_dataset.replace('noisy_', ''), 'sigma_', str(noise_level)])
                )
            else:
                save_path = None

            psnr, ssim, psnr_ens, ssim_ens = predict(model, noisy_path, label_path, device,
                                                     test_params['padding'],  n_channels, save_path)

            message = 'sigma = {} - PSNR: {:.4f} - SSIM: {:.4f} - ens PSNR: {:.4f} - ens SSIM: {:.4f}'
            print(message.format(noise_level, np.around(psnr, decimals=4), np.around(ssim, decimals=4),
                                 np.around(psnr_ens, decimals=4), np.around(ssim_ens, decimals=4)))
