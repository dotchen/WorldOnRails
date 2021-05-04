import torch
from torch import nn

class Normalize(nn.Module):
    """ ImageNet normalization """
    def __init__(self, mean, std):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std), requires_grad=False)

    def forward(self, x):
        return (x - self.mean[None,:,None,None]) / self.std[None,:,None,None]