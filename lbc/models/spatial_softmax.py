import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class SpatialSoftmax(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        
        self.height = height
        self.width = width

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self.height),
            np.linspace(-1.0, 1.0, self.width)
        )

        self.pos_x = torch.from_numpy(pos_x).reshape(-1).float()
        self.pos_x = torch.nn.Parameter(self.pos_x, requires_grad=False)

        self.pos_y = torch.from_numpy(pos_y).reshape(-1).float()
        self.pos_y = torch.nn.Parameter(self.pos_y, requires_grad=False)
    
    def forward(self, feature):
        
        flattened = feature.view(feature.shape[0], feature.shape[1], -1)
        softmax = F.softmax(flattened, dim=-1)

        expected_y = torch.sum(self.pos_y * softmax, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax, dim=-1)

        expected_xy = torch.stack([expected_x, expected_y], dim=2)

        return expected_xy