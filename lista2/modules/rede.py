import torch
import torch.nn as nn


class RedeBase(nn.Module):
    def __init__(self, sInput: int, sOutput: int) -> None:
        super().__init__()
        self.sInput = sInput
        self.sOutput = sOutput
        self.layers = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return nn.functional.log_softmax(x, dim=1)
