from typing import List, Type
from .rede import RedeBase
import torch.nn as nn


class RedeFactory:
    @staticmethod
    def createRede(
        sInput: int, sOutput: int, sLayers: List[int], fnActivation: Type[nn.Module]
    ) -> RedeBase:
        rede = RedeBase(sInput, sOutput)

        sAntes = sInput
        for sLayer in sLayers:
            rede.layers.append(nn.Linear(sAntes, sLayer))
            rede.layers.append(fnActivation())
            sAntes = sLayer
        rede.layers.append(nn.Linear(sAntes, sOutput))

        return rede
