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
        return x

    def reset(self) -> None:
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
