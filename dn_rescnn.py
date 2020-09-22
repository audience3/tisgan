import numpy as np
import torch
import torchvision
import torch.nn.init as init
import torch.nn as nn



class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        self.conv1=nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        # layers.append(nn.ReLU(inplace=True))
        for _ in range(5):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        self.block1=nn.Sequential(*layers)
        self.conv2=nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=1,padding=padding,bias=False)
        self.conv3=nn.Conv2d(in_channels=features,out_channels=features,kernel_size=1,padding=padding,bias=False)
        self.relu=nn.ReLU(inplace=True)


        self.conv4=nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False)
        # self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        residual1=self.relu(self.conv1(x))
        block1=self.block1(residual1)
        residual2=block1+residual1
        block2=self.block1(residual2)
        # residual3=block2+residual2
        # block3=self.block1(residual3)
        #layer=block3+residual3
        layer=block2+residual2
        out=self.conv4(layer)



        return out