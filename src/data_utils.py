import numpy as np
import torch

import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets


def get_dataset_loaders(dataset, batch_size, **kwargs):
    if dataset == 'fmnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data_fmnist', train=True, download=True,
                                  transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data_fmnist', train=False,
                                  transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
        class_loader = None
        
    elif dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            BinarizedMNIST('binarized_mnist_train.amat', 'data_mnist/'),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            BinarizedMNIST('binarized_mnist_test.amat', 'data_mnist/'),
            batch_size=batch_size, shuffle=True, **kwargs)
        class_loader = torch.utils.data.DataLoader(
            BinarizedMNIST('binarized_mnist_test.amat', 'data_mnist/'),
            batch_size=1, shuffle=True, **kwargs)

    return train_loader, test_loader, class_loader


class BinarizedMNIST(Dataset):
    """Binarized MNIST dataset."""

    def __init__(self, mat_file, root_dir, train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        imgs = os.path.join(root_dir, mat_file)
        self.data = np.loadtxt(imgs)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx, :].reshape((-1, 28, 28))).float()

        if self.transform:
            sample = self.transform(sample)

        # We don't return a label since it is just for the unconditional VAE setting.
        return (sample, torch.tensor(0))
