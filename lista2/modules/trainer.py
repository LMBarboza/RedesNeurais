from typing import Type
import torch.nn as nn
from torch.utils.data import DataLoader
from .treino_strategy import TreinoStrategy


class Trainer:
    def __init__(self, model: nn.Module, strategy: Type[TreinoStrategy]) -> None:
        self.model = model
        self.strategy = strategy

    def train(self, dataloader: DataLoader, epochs: int) -> None:
        self.strategy.train(self.model, dataloader, epochs)
