import torch
import torch.nn as nn
from torchvision.models import vgg11


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.base_model = vgg11()

        self.base_model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)
