# Kacper Dominiak 2020

import torch
from Test import Test
from Train import Train

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train = Train(DEVICE)

#test = Test(DEVICE)
