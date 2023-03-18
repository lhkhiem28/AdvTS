import os, sys
from libs import *

class ResNet18(nn.Module):
    def __init__(self, 
        num_classes = 7, 
    ):
        super(ResNet18, self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained = True)
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Linear(
            512, num_classes, 
        )

    def forward(self, 
        input, 
    ):
        output = self.backbone(input)

        output = self.classifier(output)
        return output