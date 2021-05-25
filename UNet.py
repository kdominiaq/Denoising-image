"""
Implementation of Resnet UNet.
"""
import torch
import torch.nn as nn


def double_conv(in_c, out_c):
    """
    convolution -> ReLU -> convolution
    :param in_c: Channel_in tensor.
    :param out_c: Channel_out tensor.
    :return: Tensor after these 3 operations.
    """
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, padding=1,kernel_size=3),
        nn.ReLU(True),
        nn.BatchNorm2d(out_c),
        nn.Conv2d(out_c, out_c, padding=1, kernel_size=3),
        nn.ReLU(True),
        nn.BatchNorm2d(out_c)
    )
    return conv


def crop_tensor(tensor, target_tensor):
    """
    Resize tensor to target_tensor.size()
    :param tensor: Tensor.
    :param target_tensor: Tensor to resize.
    :return: Tensor with resize value.
    """
    target_size = target_tensor.size()[2] # [batch_size, channels, heigth, width]
    tensor_size = tensor.size()[2]
    delta = abs(tensor_size - target_size)
    delta = delta // 2
    print(f"crop: {target_size} {tensor_size} {delta}")
    return tensor[:, :, delta:target_size - delta, delta:target_size - delta]


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop_out = nn.Dropout(0.15)

        self.down_conv_1 = double_conv(3, 112)
        self.down_conv_2 = double_conv(112, 224)
        self.down_conv_3 = double_conv(224, 448)
        self.down_conv_4 = double_conv(448, 448)

        self.up_conv_1 = double_conv(896, 224)
        self.up_conv_2 = double_conv(448, 112)
        self.up_conv_3 = double_conv(224, 112)

        self.up_1 = nn.ConvTranspose2d(in_channels=448, out_channels=448, kernel_size=2, stride=2)
        self.up_2 = nn.ConvTranspose2d(in_channels=224, out_channels=224, kernel_size=2, stride=2)
        self.up_3 = nn.ConvTranspose2d(in_channels=112, out_channels=112, kernel_size=2, stride=2)

        self.out = nn.Conv2d(in_channels=112, out_channels=3, kernel_size=1)
        self.out_2 = nn.Sigmoid()

    def forward(self, x):
        # ---- encoder ----
        # layer 1
        x1 = self.down_conv_1(x)
        x2 = self.max_pool_2x2(x1)

        # layer 2
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)

        # layer 3
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)

        # ---- decoder ----

        # layer 3
        x = self.up_1(x6)
        x = torch.cat((x, x5), dim=1)
        x = self.drop_out(x)
        x = self.up_conv_1(x)

        # layer 2
        x = self.up_2(x)
        x = torch.cat((x, x3), dim=1)
        x = self.drop_out(x)
        x = self.up_conv_2(x)

        # layer 1
        x = self.up_3(x)
        x = torch.cat((x, x1), dim=1)
        x = self.drop_out(x)
        x = self.up_conv_3(x)

        # output
        x = self.out(x)
        x = self.out_2(x)

        return x



