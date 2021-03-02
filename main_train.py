import yaml
import torch
import torch.optim as optim
from os.path import join
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from ptflops import get_model_complexity_info

from model import RDUNet
from data_management import NoisyImagesDataset, DataSampler
from train import fit_model
from transforms import AdditiveWhiteGaussianNoise, RandomHorizontalFlip, RandomVerticalFlip, RandomRot90
from utils import set_seed


def main():
    with open('config.yaml', 'r') as stream:                # Load YAML configuration file.
        config = yaml.safe_load(stream)

    model_params = config['model']
    train_params = config['train']
    val_params = config['val']

    # Defining model:
    set_seed(0)
    model = RDUNet(**model_params)

    print('Model summary:')
    test_shape = (model_params['channels'], train_params['patch size'], train_params['patch size'])
    with torch.no_grad():
        macs, params = get_model_complexity_info(model, test_shape, as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # Define the model name and use multi-GPU if it is allowed.
    model_name = 'model_color' if model_params['channels'] == 3 else 'model_gray'
    device = torch.device(train_params['device'])
    print("Using device: {}".format(device))
    if torch.cuda.device_count() > 1 and 'cuda' in device.type and train_params['multi gpu']:
        model = nn.DataParallel(model)
        print('Using multiple GPUs')

    model = model.to(device)
    param_group = []
    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            p = {'params': param, 'weight_decay': train_params['weight decay']}
        else:
            p = {'params': param, 'weight_decay': 0.}
        param_group.append(p)

    # Load training and validation file names.
    # Modify .txt files if datasets do not fit in memory.
    with open('train_files.txt', 'r') as f_train, open('val_files.txt', 'r') as f_val:
        raw_train_files = f_train.read().splitlines()
        raw_val_files = f_val.read().splitlines()
        train_files = list(map(lambda file: join(train_params['dataset path'], file), raw_train_files))
        val_files = list(map(lambda file: join(val_params['dataset path'], file), raw_val_files))

    training_transforms = transforms.Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRot90()
    ])

    # Predefined noise level
    train_noise_transform = [AdditiveWhiteGaussianNoise(train_params['noise level'], clip=True)]
    val_noise_transforms = [AdditiveWhiteGaussianNoise(s, fix_sigma=True, clip=True) for s in val_params['noise levels']]

    print('\nLoading training dataset:')
    training_dataset = NoisyImagesDataset(train_files,
                                          model_params['channels'],
                                          train_params['patch size'],
                                          training_transforms,
                                          train_noise_transform)

    print('\nLoading validation dataset:')
    validation_dataset = NoisyImagesDataset(val_files,
                                            model_params['channels'],
                                            val_params['patch size'],
                                            None,
                                            val_noise_transforms)
    # Training in sub-epochs:
    print('Training patches:', len(training_dataset))
    print('Validation patches:', len(validation_dataset))
    n_samples = len(training_dataset) // train_params['dataset splits']
    n_epochs = train_params['epochs'] * train_params['dataset splits']
    sampler = DataSampler(training_dataset, num_samples=n_samples)

    data_loaders = {
        'train': DataLoader(training_dataset, train_params['batch size'], num_workers=train_params['workers'], sampler=sampler),
        'val': DataLoader(validation_dataset, val_params['batch size'], num_workers=val_params['workers']),
    }

    # Optimization:
    learning_rate = train_params['learning rate']
    step_size = train_params['scheduler step'] * train_params['dataset splits']

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(param_group, lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=train_params['scheduler gamma'])

    # Train the model
    fit_model(model, data_loaders, model_params['channels'], criterion, optimizer, lr_scheduler, device,
              n_epochs, val_params['frequency'], train_params['checkpoint path'], model_name)


if __name__ == '__main__':
    main()
