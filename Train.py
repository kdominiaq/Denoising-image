"""
Module inherits parameters from Params class. It is responsible for training model by one of structure (Unet, Resnet and
Autoencoder). It saves models from given path. Module uses MSELoss for criterion test, Adam's optimizer and "salt and pepper"
(s&p) for noising images before testing.

To pick which type of structure is using, must be change value in line __ (model = torch. load..), folder's path to save
images (image_save(...)) in line 50 and dataset in line  53.

Module allows to save image during the learning. Variable from Params class: "self.saving_image_during_learning" must be True.

"""

import torch
from Dataset import Dataset
from torch import nn
from skimage.util import random_noise
from torchvision.utils import save_image

from Autoencoder import Autoencoder
from UNet import UNet
from Resnet import Resnet

from Params import Params


def image_save(img, name):
    """
    Saves the pics under the given path.
    :param img: tensor size -> must be torch.Size([1, 3, 32, 96]).
    :param name: path name.
    :return: none.
    """
    # -1 -> when we don't know about size of width/height -> must be 96
    img = img.view(img.size(0), 3, 256, -1)
    save_image(img, name)


class Train(Params):
    """
    Inherited class of Params. It trains module by one of picked structure (Unet, Resnet and Autoencoder) and also
    allows to saving images and trained module.
    """
    def __init__(self, device):
        super().__init__()
        self.clean_img_train = None
        self.noised_img_train = None
        self.output = None
        self.loss = None
        model = Resnet().cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        train_load = Dataset.CIFAR_train_loader()
        for epoch in range(self.num_epochs):
            for i, data in enumerate(train_load):
                self.clean_img_train, _ = data[0], data[1]
                self.noised_img_train = torch.tensor(random_noise(self.clean_img_train,
                                                                  mode='s&p',
                                                                  salt_vs_pepper=0.5,
                                                                  clip=True))
                random_noise(self.clean_img_train, mode='s&p', salt_vs_pepper=0.5, clip=True)
                # fixing "RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same"
                self.clean_img_train, self.noised_img_train = self.clean_img_train.to(device), self.noised_img_train.to(device)
                output = model(self.noised_img_train)
                self.loss = criterion(output, self.clean_img_train)
                # backward
                self.loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # period of the number of photos in dateset
                if i == self.num_train_images_in_epoch:
                    break
                # log
            print(f'epoch [{epoch + 1}/{self.num_epochs}], loss:{self.loss.item():.4f}')

            # saving image during learning
            if epoch % 10 == 0 and self.saving_image_during_learning:
                combined_img = torch.cat((self.clean_img_train, self.noised_img_train, self.output), 3) #combining images
                image_save(combined_img, f"./{epoch}_all_vs1.png")

        torch.save(model, self.Resnet_model_save_PATH)
