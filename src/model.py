import torch
import torch.nn as nn
from torchvision import models

class DeepfakeResNet(nn.Module):
    def __init__(self):
        super(DeepfakeResNet, self).__init__()

        self.model = models.resnet18(pretrained=True)

  
        for param in self.model.parameters():
            param.requires_grad = False

       
        for param in self.model.layer4.parameters():
            param.requires_grad = True

 
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.model(x)