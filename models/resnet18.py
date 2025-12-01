# resnet18 model adapted for CIFAR-10 and ImageNet-like images using resnet18 from torchvision

import torch
import torch.nn as nn
from torchvision.models import resnet18


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        args={"num_classes":10, "input_size":32, "dropout":0.0}
        self.model = resnet18()#weights='IMAGENET1K_V1') #loads imagenet weights

        if args['input_size'] <= 64:
            # Modify first conv and remove maxpool for small images like CIFAR
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.model.maxpool = nn.Identity()

        # Optionally add dropout before final FC
        self.dropout = nn.Dropout(p=args['dropout']) if args['dropout'] > 0 else nn.Identity()

        # Replace the classifier
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, args['num_classes'])

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.model.fc(x)
        return x

if __name__ == '__main__':
    model = Net()

    x = torch.randn(1, 3, 32, 32)
    print(model(x).shape) 

