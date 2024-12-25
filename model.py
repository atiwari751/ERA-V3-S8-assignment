from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 32x32 | RF = 3

        # Convolution Block 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # output_size = 32x32 | RF = 5
        
        # Transition Block 1
        self.trans1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),    # output_size = 16x16 | RF = 6
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 16x16 | RF = 6

        # Convolution Block 2 (Depthwise Separable)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # output_size = 16x16 | RF = 10
        
        # Transition Block 2
        self.trans2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),    # output_size = 8x8 | RF = 12
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 8x8 | RF = 12

        # Convolution Block 3 (with dilation)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, dilation=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # output_size = 8x8 | RF = 36

        # Transition Block 3
        self.trans3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),    # output_size = 4x4 | RF = 40
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 4x4 | RF = 40

        # Convolution Block 4
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # output_size = 4x4 | RF = 56

        # Output Block
        self.gap = nn.AdaptiveAvgPool2d(1)  # output_size = 1x1 | RF = 56
        self.linear = nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)      # 32x32 | RF = 3
        x = self.conv2(x)      # 32x32 | RF = 5
        x = self.trans1(x)     # 16x16 | RF = 6
        x = self.conv3(x)      # 16x16 | RF = 10
        x = self.trans2(x)     # 8x8   | RF = 12
        x = self.conv4(x)      # 8x8   | RF = 36
        x = self.trans3(x)     # 4x4   | RF = 40
        x = self.conv5(x)      # 4x4   | RF = 56
        x = self.gap(x)        # 1x1   | RF = 56
        x = self.linear(x)     # 1x1   | RF = 56
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

if __name__ == '__main__':
    device = get_device()
    print(device)
    model = Net().to(device)
    summary(model, input_size=(3, 32, 32))