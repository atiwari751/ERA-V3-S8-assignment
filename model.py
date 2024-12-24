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
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 32 | RF = 3

        # Convolution Block 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 32 | RF = 5
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 32 | RF = 7
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 32 | RF = 9

        # Transition Block 1
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, bias=False),
            nn.ReLU()
        ) # output_size = 16 | RF = 10
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=16, kernel_size=1, bias=False),
            nn.ReLU()
        ) # output_size = 16 | RF = 10

        # Convolution Block 2
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 16 | RF = 14
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 16 | RF = 18
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 16 | RF = 22

        # Transition Block 2
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, bias=False),
            nn.ReLU()
        ) # output_size = 8 | RF = 24
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=16, kernel_size=1, bias=False),
            nn.ReLU()
        ) # output_size = 8 | RF = 24

        # Convolution Block 3
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 8 | RF = 32
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 8 | RF = 40
        self.conv14 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 8 | RF = 48

        # Transition Block 3
        self.conv15 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, bias=False),
            nn.ReLU()
        ) # output_size = 4 | RF = 52
        self.conv16 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=16, kernel_size=1, bias=False),
            nn.ReLU()
        ) # output_size = 4 | RF = 52

        # Convolution Block 4
        self.conv17 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 4 | RF = 68
        self.conv18 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 4 | RF = 84
        self.conv19 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 4 | RF = 100

        # Output Block
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output_size = 1 | RF = 100
        
        self.linear = nn.Conv2d(in_channels=128, out_channels=10, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x = self.conv19(x)
        x = self.gap(x)
        x = self.linear(x)
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