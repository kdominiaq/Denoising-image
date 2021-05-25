"""
Module contains all dateset which can be use:
- CIFAR,
- my_test/train_loader: images 32x32.
- my_256_test_train_loader: images 256x256.
"""

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


class Dataset:
    """
    Dateset: CIFAR10 and private if exist.
    transform and return datasets -> test, train
    """
    def __init__(self):
        super(Dataset, self).__init__()
        self._transform = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def CIFAR_train_loader(self):
        """
        :return: train_loader
        """
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self._transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)
        return train_loader

    def CIFAR_test_loader(self):
        """
        :return: test_loader
        """
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self._transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=2)
        return test_loader

    def my_test_loader(self):
        """
        :return: my_test_loader
        """
        my_test_set = torchvision.datasets.ImageFolder(root='./test_image', transform=self._transform)
        my_test_loader = torch.utils.data.DataLoader(my_test_set, batch_size=1, shuffle=False)
        return my_test_loader

    def my_256_train_loader(self):
        """
        :return: my_256_train_loader
        """
        my_train_set = torchvision.datasets.ImageFolder(root='~/datasets/Train_256_256', transform=self._transform)
        my_train_loader = torch.utils.data.DataLoader(my_train_set, batch_size=1, shuffle=False)
        return my_train_loader

    def my_256_test_loader(self):
        """
        :return: my_256_test_loader
        """
        my_test_set = torchvision.datasets.ImageFolder(root='~/datasets/Test_256_256', transform=self._transform)
        my_test_loader = torch.utils.data.DataLoader(my_test_set, batch_size=1, shuffle=False)
        return my_test_loader


Dataset = Dataset()
