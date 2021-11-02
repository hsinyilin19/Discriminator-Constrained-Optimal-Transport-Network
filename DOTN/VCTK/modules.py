import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.blstm = nn.LSTM(257, 1024, dropout=0.0, num_layers=2, bidirectional=True, batch_first=True)
        self.LReLU = nn.LeakyReLU(0.3)
        self.ReLU = nn.ReLU()
        self.Dropout = nn.Dropout(p=0.0)
        self.fc1 = nn.Linear(1024 * 2, 1024)
        self.fc2 = nn.Linear(1024, 257)

    def forward(self, x):
        #  x: clean mag, y: noise mag
        output, _ = self.blstm(x)
        output = self.fc1(output)
        output = self.LReLU(output)
        output = self.Dropout(output)
        output = self.fc2(output)
        output = self.ReLU(output)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(  # input shape (batch_size, 1, 64, 257)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=8,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (batch_size, 16, 32, 128)
        )
        self.conv2 = nn.Sequential(  # input shape (batch_size, 16, 32, 128)
            nn.Conv2d(8, 16, 5, 1, 2),
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (batch_size, 32, 16, 64)
        )
        self.out1 = nn.Sequential(
            nn.Linear(16 * 16 * 64, 16 * 16),  # fully connected layer, output 10 classes
            nn.ReLU()  # activation
        )

        self.out2 = nn.Sequential(
            nn.Linear(16 * 16, 1),  # fully connected layer, output 10 classes
            nn.Sigmoid()  # activation
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 16 * 64)
        x = self.out1(x)
        output = self.out2(x)
        return output