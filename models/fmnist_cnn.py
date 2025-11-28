"""
CNN for FMNIST (28*28*1, 10 classses)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self) -> None:
        args = {"num_classes": 10}
        super(Net, self).__init__()
        self.NUM_CLASSES = args["num_classes"]
        print("FMNIST_CNN constructor called.")

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # output of conv2 has 50 channels and result of pool2 would be 4*4 so its 50*4*4
        self.fc1 = nn.Linear(in_features=50 * 4 * 4, out_features=800)
        self.fc2 = nn.Linear(in_features=800, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=10)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # flatten
        x = x.view(-1, 50 * 4 * 4)  # 50 channels * 4x4 spatial dimensions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    net = Net()
    print(net(torch.randn(1, 1, 28, 28)))

