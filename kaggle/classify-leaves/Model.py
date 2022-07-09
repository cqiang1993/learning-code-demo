import torch.nn as nn
from torchvision import datasets, models, transforms

class MODEL(nn.Module):
    def __init__(self, out_label):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, out_label)

    def forward(self, X):
        return self.resnet(X)