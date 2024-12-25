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
        ) # output_size = 32x32 | RF = 3 (j_in=1, RF = 1 + (3-1)*1 = 3)

        # Convolution Block 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # output_size = 32x32 | RF = 5 (j_in=1, RF = 3 + (3-1)*1 = 5)
        
        # Transition Block 1
        self.trans1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),    # output_size = 16x16 | RF = 6 (j_in=1, RF = 5 + (2-1)*1 = 6)
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 16x16 | RF = 6 (1x1 doesn't change RF)

        # Convolution Block 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # output_size = 16x16 | RF = 10 (j_in=2, RF = 6 + (3-1)*2 = 10)
        
        # Transition Block 2
        self.trans2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),    # output_size = 8x8 | RF = 12 (j_in=2, RF = 10 + (2-1)*2 = 12)
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 8x8 | RF = 12 (1x1 doesn't change RF)

        # Convolution Block 3
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # output_size = 8x8 | RF = 20 (j_in=4, RF = 12 + (3-1)*4 = 20)

        # Transition Block 3
        self.trans3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),    # output_size = 4x4 | RF = 24 (j_in=4, RF = 20 + (2-1)*4 = 24)
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 4x4 | RF = 24 (1x1 doesn't change RF)

        # Convolution Block 4
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # output_size = 4x4 | RF = 40 (j_in=8, RF = 24 + (3-1)*8 = 40)

        # Output Block
        self.gap = nn.AdaptiveAvgPool2d(1)  # output_size = 1x1 | RF = 40 (GAP doesn't change RF)
        self.linear = nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)      # 32x32 | RF = 3
        x = self.conv2(x)      # 32x32 | RF = 5
        x = self.trans1(x)     # 16x16 | RF = 6
        x = self.conv3(x)      # 16x16 | RF = 10
        x = self.trans2(x)     # 8x8   | RF = 12
        x = self.conv4(x)      # 8x8   | RF = 20
        x = self.trans3(x)     # 4x4   | RF = 24
        x = self.conv5(x)      # 4x4   | RF = 40
        x = self.gap(x)        # 1x1   | RF = 40
        x = self.linear(x)     # 1x1   | RF = 40
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