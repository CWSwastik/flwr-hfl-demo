# vgg16_model.py

import torch
import torch.nn as nn
from torchvision.models import vgg16


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        args = {"num_classes": 10, "input_size": 32, "dropout": 0.0}
        # Load VGG16 convolutional backbone without any imagenet weights
        self.features = vgg16().features

        # Compute the output size after conv layers
        dummy_input = torch.zeros(1, 3, args["input_size"], args["input_size"])
        with torch.no_grad():
            feature_map = self.features(dummy_input)
            feature_dim = feature_map.view(1, -1).size(1)

        self.flatten_dim = feature_dim

        # Choose classifier based on input size
        if args["input_size"] >= 224:
            # Original VGG16 classifier (for ImageNet-like images)
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 4096),
                nn.ReLU(True),
                nn.Dropout(p=args["dropout"]),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(p=args["dropout"]),
                nn.Linear(4096, args["num_classes"]),
            )
        else:
            # Lighter classifier for small images like CIFAR-10
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(True),
                nn.Dropout(p=args["dropout"]),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(p=args["dropout"]),
                nn.Linear(512, args["num_classes"]),
            )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
