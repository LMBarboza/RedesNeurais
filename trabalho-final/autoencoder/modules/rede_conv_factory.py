from typing import List, Type
from .rede import AutoEncoder
from .abstract_rede_factory import AbstractRedeFactory
import torch.nn as nn


class ConvAutoencoderFactory(AbstractRedeFactory):
    def createRede(
        self,
        sInput: int,
        sLatent: int,
        sLayers: List[int],
        fnActivation: Type[nn.Module],
        sKernel: int,
        dropout: bool = False,
    ) -> AutoEncoder:

        rede = AutoEncoder(sLatent)
        encoder = nn.Sequential()
        decoder = nn.Sequential()
        sAntes = sInput
        sLinearInput = 28

        for sLayer in sLayers:
            encoder.add_module(
                f"conv_{sAntes}_{sLayer}", nn.Conv2d(sAntes, sLayer, sKernel)
            )
            encoder.add_module(f"act_{sLayer}", fnActivation())
            sLinearInput = (sLinearInput - sKernel) + 1
            if dropout:
                encoder.add_module(f"dropout_{sLayer}", nn.Dropout(0.25))
            sAntes = sLayer

        sLinearInput = sLinearInput**2 * sAntes
        encoder.add_module("flatten", nn.Flatten())
        encoder.add_module("linear_latente", nn.Linear(sLinearInput, sLatent))

        h = w = int((sLinearInput // sAntes) ** 0.5)
        decoder.add_module("latente_linear", nn.Linear(sLatent, sLinearInput))
        decoder.add_module(
            "unflatten",
            nn.Unflatten(1, (sAntes, h, w)),
        )

        for i, sLayer in enumerate(reversed(sLayers)):
            if i == len(sLayers) - 1:
                sLayer = 1

            decoder.add_module(
                f"conv_{sAntes}_{sLayer}", nn.ConvTranspose2d(sAntes, sLayer, sKernel)
            )
            decoder.add_module(f"act_{sLayer}", fnActivation())
            sAntes = sLayer

        rede.encoder = encoder
        rede.decoder = decoder

        return rede
