import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


class Dataset:
    '''
    Dateset: CIFAR10
    transform and return datasets -> test, train
    '''

    def __init__(self):
        super(Dataset, self).__init__()
        self._transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def train_loader(self):
        '''
        :return: train_loader
        '''

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self._transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)
        return train_loader

    def test_loader(self):
        '''
        :return: test_loader
        '''

        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self._transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=2)
        return test_loader


Dataset = Dataset()
