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
    ) -> List[float]:
        pass

    @abstractmethod
    def evaluate(
        self, model: nn.Module, dataloader: DataLoader, device: torch.device
    ) -> float:
        pass


class VAEStrategy(TreinoStrategy):
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
    ) -> List[float]:

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for _, (inputs, targets) in enumerate(dataloader):
                self.optimizer.zero_grad()
                inputs = inputs.view(-1, 28 * 28)
                inputs = inputs.to(device)
                recon, mu, logvar = model(inputs)
                loss = self.fnLoss(recon, inputs, mu, logvar)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        return []

    def evaluate(
        self, model: nn.Module, dataloader: DataLoader, device: torch.device
    ) -> float:
        return 0.0


class GenerateStrategy(TreinoStrategy):
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
    ) -> List[float]:

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for _, (inputs, targets) in enumerate(dataloader):
                self.optimizer.zero_grad()
                if model.mlp:
                    inputs = inputs.view(-1, 28 * 28)
                inputs = inputs.to(device)
                _, outputs = model(inputs)
                loss = self.fnLoss(outputs, inputs)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        return []

    def evaluate(
        self, model: nn.Module, dataloader: DataLoader, device: torch.device
    ) -> float:
        return 0.0


class ClassificationStrategy(TreinoStrategy):
    def __init__(self, fnLoss: List[nn.Module], optimizer: optim.Optimizer) -> None:
        self.fnLoss = fnLoss
        self.optimizer = optimizer

    def train(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        testDataloader: DataLoader,
        epochs: int,
        device: torch.device,
    ) -> List[float]:

        accuracy_list = []
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for _, (inputs, targets) in enumerate(dataloader):
                self.optimizer.zero_grad()
                if model.mlp:
                    inputs = inputs.view(-1, 28 * 28)
                inputs, targets = inputs.to(device), targets.to(device)
                input_encoded, outputs = model(inputs)
                loss_classification = self.fnLoss[0](input_encoded, inputs)
                loss_decoder = self.fnLoss[1](outputs, inputs)
                loss = loss_classification + loss_decoder
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            accuracy = self.evaluate(model, testDataloader, device)
            accuracy_list.append(accuracy)
            print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {accuracy:.2f}%")

        return accuracy_list

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
                if model.mlp:
                    inputs = inputs.view(-1, 28 * 28)
                input_encoded, _ = model(inputs)
                _, predicted = torch.max(input_encoded.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        return accuracy
