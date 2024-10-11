import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet18', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)