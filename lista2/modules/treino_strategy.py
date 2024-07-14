from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class TreinoStrategy(ABC):
    @abstractmethod
    def train(self, model: nn.Module, dataloader: DataLoader, epochs: int) -> None:
        pass


class STDStrategy(TreinoStrategy):
    def __init__(self, fnLoss: nn.Module, optimizer: optim.Optimizer) -> None:
        self.fnLoss = fnLoss
        self.optimizer = optimizer

    def train(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        epochs: int,
        device: torch.device,
    ) -> None:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for _, (inputs, targets) in enumerate(dataloader):
                self.optimizer.zero_grad()
                inputs = inputs.view(-1, 28 * 28)
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = self.fnLoss(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
