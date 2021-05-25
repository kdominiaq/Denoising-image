"""
Module inherits parameters from Params class. It is responsible for testing trained models which has been trained before.
It downloads models from given path and tests its possibilities on specially separated images. Module uses MSELoss for
criterion test and "salt and pepper" (s&p) for noising images before testing.

To pick which type of structure is using  (Unet, Resnet and Autoencoder), must be change value in line 41, dataset in
(model = torch.load..) and folder's path to save images (image_save(...)) in line 57. Dataset can be changed in line 44.

Module allows to save image  during the test. Variable from Params class: "self.saving_image_during_testing" must be True.
"""

import torch
from torchvision.utils import save_image
from Dataset import Dataset
from torch import nn
from skimage.util import random_noise
from Params import Params


def image_save(img, name):
    """
    Saves the pics under the given path name.
    :param img: tensor size -> must be torch.Size([1, 3, 32, 96]).
    :param name: path name.
    :return: none.
    """
    # -1 -> when we don't know about size of width/height -> must be 96
    img = img.view(img.size(0), 3, 256, -1)
    save_image(img, name)


# test
class Test(Params):
    def __init__(self, device):
        super().__init__()
        self.clean_img_test = None
        self.noised_img_test = None
        self.test_output = None
        self._device = device
        self._criterion_test = nn.MSELoss()
        model = torch.load(self.Resnet_model_save_PATH)
        model.eval()
        loss = 0
        for i, data in enumerate(Dataset.CIFAR_test_loader()):
            # prepare test dataset
            self.clean_img_test, _ = data[0], data[1]
            self.noised_img_test = torch.tensor(random_noise(self.clean_img_test,
                                                             mode='s&p',
                                                             salt_vs_pepper=0.5,
                                                             clip=True))
            self.clean_img_test, self.noised_img_test = self.clean_img_test.to(device), self.noised_img_test.to(device)

            self.test_output = model(self.noised_img_test)
            _, predicted = torch.max(self.test_output.data, 1)

            loss += self._criterion_test(self.test_output, self.clean_img_test).item()

            if i == self.num_test_images - 1:
                average_loss_test = loss / self.num_test_images
                print(f'Average loss: {average_loss_test:.8f}')
                if self.saving_image_during_testing:
                    combined_img = torch.cat((self.clean_img_test, self.noised_img_test, self.test_output), 3)
t