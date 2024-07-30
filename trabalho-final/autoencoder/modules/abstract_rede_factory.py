from abc import ABC, abstractmethod
from typing import List, Type
import torch.nn as nn
from .rede import AutoEncoder


class AbstractRedeFactory(ABC):
    @abstractmethod
    def createRede(
        self,
        sInput: int,
        sLatent: int,
        sLayers: List[int],
        fnActivation: Type[nn.Module],
        sKernel: int = 3,
        dropout: bool = False,
    ) -> AutoEncoder:
        pass
