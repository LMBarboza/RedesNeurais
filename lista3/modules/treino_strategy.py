from abc import ABC, abstractmethod
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class TreinoStrategy(ABC):
    @abstractmethod
    def train(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        testDataloader: DataLoader,
        epochs: int,
        device: torch.device,
    ) -> tuple[List[float], List[float]]:
        pass

    @abstractmethod
    def evaluate(
        self, model: nn.Module, dataloader: DataLoader, device: torch.device
    ) -> float:
        pass


class STDStrategy(TreinoStrategy):
    def __init__(self, fnLoss: nn.Module, optimizer: optim.Optimizer) -> None:
        self.fnLoss = fnLoss
        self.optimizer = optimizer

    def train(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        testDataloader: DataLoader,
        epochs: int,
        device: torch.device,
    ) -> tuple[List[float], List[float]]:

        accuracy_list = []
        train_accuracy_list = []
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            for _, (inputs, targets) in enumerate(dataloader):
                self.optimizer.zero_grad()
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = self.fnLoss(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            train_accuracy = 100 * correct / total
            train_accuracy_list.append(train_accuracy)
            print(f"Epoch {epoch+1}/{epochs}, Train Accuracy: {train_accuracy:.2f}%")
            accuracy = self.evaluate(model, testDataloader, device)
            accuracy_list.append(accuracy)
            print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {accuracy:.2f}%")

        return accuracy_list, train_accuracy_list

    def evaluate(
        self, model: nn.Module, dataloader: DataLoader, device: torch.device
    ) -> float:
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        return accuracy
