import numpy as np

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets

from utils.transform import training_transform, validation_transform


def create_loaders(model, config, device='cpu'):
    """
    Create data loaders
    """
    # Get configurations
    data_dir = config['data']['data_dir']
    shuffle = config['data']['shuffle']
    validation_split = config['data']['validation_split']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    seed = config['data']['random_seed']

    if device == 'cuda' and torch.cuda.is_available():
        pin_memory = True
    else:
        pin_memory = False
    
    # Load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, 
        download=True, transform=training_transform
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True, 
        download=False, transform=training_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, 
        download=False, transform=training_transform
    )

    # Split the training data into training and validation
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_split * num_train))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Create the loaders 
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        sampler=valid_sampler, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, valid_loader, test_loader
