from typing import List, Type
from .rede import RedeBase
import torch.nn as nn


class RedeFactory:
    @staticmethod
    def createRede(
        sChannels: int,
        sOutput: int,
        sLayers: List[int],
        sKernel: int,
        fnActivation: Type[nn.Module],
        dropout: bool = False,
    ) -> RedeBase:
        rede = RedeBase()

        sAntes = sChannels
        sLinearInput = 32
        for sLayer in sLayers:
            rede.layers.append(nn.Conv2d(sAntes, sLayer, sKernel))
            rede.layers.append(fnActivation())
            sLinearInput = (sLinearInput - sKernel) + 1
            if dropout:
                rede.layers.append(nn.Dropout(0.25))
            sAntes = sLayer

        sLinearInput = sLinearInput**2 * sAntes
        rede.layers.append(nn.Flatten())
        rede.layers.append(nn.Linear(sLinearInput, 120))
        rede.layers.append(fnActivation())
        rede.layers.append(nn.Linear(120, 84))
        rede.layers.append(fnActivation())
        rede.layers.append(nn.Linear(84, sOutput))
        return rede
