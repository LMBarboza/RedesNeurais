from typing import List, Type
from .rede import AutoEncoder
from .abstract_rede_factory import AbstractRedeFactory
import torch.nn as nn


class MLPAutoencoderFactory(AbstractRedeFactory):
    def createRede(
        self,
        sInput: int,
        sLatent: int,
        sLayers: List[int],
        fnActivation: Type[nn.Module],
        sKernel,
        dropout,
    ) -> AutoEncoder:
        rede = AutoEncoder(sLatent, mlp=True)
        encoder = nn.Sequential()
        decoder = nn.Sequential()
        sAntes = sInput

        for sLayer in sLayers:
            encoder.add_module(f"linear_{sAntes}_{sLayer}", nn.Linear(sAntes, sLayer))
            encoder.add_module(f"act_{sLayer}", fnActivation())
            sAntes = sLayer
        encoder.add_module("linear_latente", nn.Linear(sAntes, sLatent))

        sAntes = sLatent
        for sLayer in reversed(sLayers):
            decoder.add_module(f"linear_{sAntes}_{sLayer}", nn.Linear(sAntes, sLayer))
            decoder.add_module(f"act_{sLayer}", fnActivation())
            sAntes = sLayer
        decoder.add_module("latente_linear", nn.Linear(sAntes, sInput))

        rede.encoder = encoder
        rede.decoder = decoder

        return rede
