from typing import Type, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .treino_strategy import TreinoStrategy


class Trainer:
    def __init__(self, model: nn.Module, strategy: Type[TreinoStrategy]) -> None:
        self.model = model
        self.strategy = strategy

    def train(
        self,
        dataloader: DataLoader,
        testDataloader: DataLoader,
        epochs: int,
        device: torch.device,
    ) -> List[float]:
        return self.strategy.train(
            self.model, dataloader, testDataloader, epochs, device
        )
