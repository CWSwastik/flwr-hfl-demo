import torch
import torch.nn as nn
from torchvision.models import vgg11


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        # Load the VGG11 architecture without pretrained weights
        self.base_model = vgg11()

        # Modify the first conv layer if needed (optional, but here it's left unchanged as 3x3 works fine)
        # self.base_model.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        # Adjust classifier to output 10 classes for CIFAR-10
        self.base_model.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        # Remove AdaptiveAvgPool since CIFAR-10 images are smaller and result in 1x1 feature maps after pooling
        self.base_model.avgpool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)
