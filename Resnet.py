"""
Implementation of Resnet Module.
"""
import torch
import torch.nn as nn
import torchvision.models as models


def double_conv(in_c, out_c):
    """
    convolution -> ReLU -> convolution
    :param in_c: Channel_in tensor.
    :param out_c: Channel_out tensor.
    :return: Tensor after these 3 operations
.    """
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, padding=1,kernel_size=3),
        nn.ReLU(True),
        nn.BatchNorm2d(out_c),
        nn.Conv2d(out_c, out_c, padding=1, kernel_size=3),
        nn.ReLU(True),
        nn.BatchNorm2d(out_c)
    )
    return conv


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

        self.drop_out = nn.Dropout(0.15)

        self.resnet_1 = torch.nn.Sequential(*(list(self.resnet.children())[:-10])) # [1, 3, 256, 256]
        self.resnet_2 = torch.nn.Sequential(*(list(self.resnet.children())[:-7])) # [1, 64, 128, 128]
        self.resnet_4 = torch.nn.Sequential(*(list(self.resnet.children())[:-5])) # [1, 256, 64, 64]
        self.resnet_5 = torch.nn.Sequential(*(list(self.resnet.children())[:-4])) # [1, 512, 32, 32]

        self.up_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=2, stride=2)
        self.up_3 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=2, stride=2)

        self.up_conv_1 = double_conv(512,256)
        self.up_conv_2 = double_conv(128,64)
        self.up_conv_3 = double_conv(6,3)

        self.out_2 = nn.Sigmoid()

    def forward(self, x):
        x1 = self.resnet_1(x)
        x2 = self.resnet_2(x)
        x4 = self.resnet_4(x)
        x = self.resnet_5(x)

        x = self.up_1(x)
        x = torch.cat((x, x4), dim=1)
        x = self.drop_out(x)
        x = self.up_conv_1(x)

        x = self.up_2(x)
        x = torch.cat((x, x2), dim=1)
        x = self.drop_out(x)
        x = self.up_conv_2(x)

        x = self.up_3(x)
        x = torch.cat((x, x1), dim=1)
        x = self.drop_out(x)
        x = self.up_conv_3(x)

        x = self.out_2(x)

        return x









