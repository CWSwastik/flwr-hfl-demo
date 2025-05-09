import torch
import torch.nn as nn
from torchvision.models import vgg11


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.base_model = vgg11()

        # self.base_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # # self.base_model.classifier = nn.Sequential(
        # #     nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        # # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)
