import torch
import torch.nn as nn

def double_conv(in_c, out_c):
    '''
    convolution -> ReLU -> convolution
    :param in_c: channel_in tensor
    :param out_c: channel_out tensor
    :return: tensor after these 3 operations
    '''
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
    '''
    resize tensor to target_tensor.size()
    :param tensor: tensor
    :param target_tensor: tensor to resize
    :return: tensor with resize value
    '''
    target_size = target_tensor.size()[2] # [batch_size, channels, heigth, width], czyli bierzemy element height
    tensor_size = tensor.size()[2]
    delta = abs(tensor_size - target_size)
    delta = delta //2
    print(f"crop: {target_size} {tensor_size} {delta}")
    return tensor[:, :, delta:target_size - delta , delta:target_size - delta ]


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
        # encoder

        # input
        #print(x.size())

        # layer 1
        x1 = self.down_conv_1(x)
        #print(x1.size())
        x2 = self.max_pool_2x2(x1)
        #print(x2.size())

        # layer 2
        x3 = self.down_conv_2(x2)
        #print(x3.size())
        x4 = self.max_pool_2x2(x3)
        #print(x4.size())

        # layer 3
        x5 = self.down_conv_3(x4)
        #print(x5.size())
        x6 = self.max_pool_2x2(x5)
        #print(x6.size())

        # decoder

        # layer 3
        x = self.up_1(x6)
        #print(x.size())
        x = torch.cat((x, x5), dim=1)
        #print(x.size())
        x = self.drop_out(x)
        #print(x.size())
        x = self.up_conv_1(x)
        #print(x.size())

        # layer 2
        x = self.up_2(x)
        #print(x.size())
        x = torch.cat((x, x3), dim=1)
        #print(x.size())
        x = self.drop_out(x)
        #print(x.size())
        x = self.up_conv_2(x)
        #print(x.size())

        # layer 1
        x = self.up_3(x)
        #print(x.size())
        x = torch.cat((x, x1), dim=1)
        #print(x.size())
        x = self.drop_out(x)
        #print(x.size())
        x = self.up_conv_3(x)
        #print(x.size())

        # output
        x = self.out(x)
        #print(x.size())
        x =self.out_2(x)

        return x



