import numpy as np
import torch.nn as nn
import torch

# https://towardsdatascience.com/text-classification-with-cnns-in-pytorch-1113df31e79f


class Cnn(nn.Module):
    def __init__(self, num_class=2) -> None:
        super(Cnn, self).__init__()
        # Check here for GPU support https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))

        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

        # Convolution layers setup
        self.conv_layer1 = nn.LazyConv1d(out_channels=200, kernel_size=5)
        self.conv_layer2 = nn.Conv1d(
            in_channels=200, out_channels=100, kernel_size=5)
        self.conv_layer3 = nn.Conv1d(
            in_channels=100, out_channels=25, kernel_size=5)

        # Max pooling layers
        self.max_pool1 = nn.MaxPool1d(kernel_size=5, stride=2)
        self.max_pool2 = nn.MaxPool1d(kernel_size=5, stride=2)
        self.max_pool3 = nn.MaxPool1d(kernel_size=5, stride=2)

        # Full connected layer
        self.fc = nn.LazyLinear(out_features=num_class)

    def forward(self, x):
        # in_channels: number of input
        # out_channels: number of output (which is inpute of next layer)
        # in_channels: size of sliding window on input data
        out = self.conv_layer1(x)
        out = torch.relu(out)
        out = self.max_pool1(out)

        out = self.conv_layer2(out)
        out = torch.relu(out)
        out = self.max_pool2(out)

        out = self.conv_layer3(out)
        out = torch.relu(out)
        out = self.max_pool3(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc(out)
        out = torch.sigmoid(out)

        return out.squeeze()
