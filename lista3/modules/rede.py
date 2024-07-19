import torch
import torch.nn as nn


class RedeBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
