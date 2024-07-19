import argparse
import json
import configparser
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from modules.rede_factory import RedeFactory


def evaluate_ensemble(
    model_list: List[nn.Module], dataloader: DataLoader, device: torch.device
) -> float:

    for model in model_list:
        model.to(device)
        model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(-1, 28 * 28)
            ensemble_outputs = torch.zeros(inputs.size(0), 10).to(device)
            for model in model_list:
                outputs = model(inputs)
                ensemble_outputs += outputs

            ensemble_outputs /= len(model_list)

            _, predicted = torch.max(ensemble_outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    print(accuracy)
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinamento MLP")
    parser.add_argument(
        "--config",
        type=str,
        default="configurations.ini",
        help="PATH para configurações de treino",
    )

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    batch_size = config.getint("TRAINING", "batch_size")
    use_cuda = config.getboolean("TRAINING", "cuda")

    cuda = use_cuda and torch.cuda.is_available()

    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    sInput = 28 * 28
    sOutput = 10

    hiddenLayers = [[], [128], [128, 64], [128, 64, 32]]

    model_list = []

    for i, layers in enumerate(hiddenLayers):
        model = RedeFactory.createRede(
            sInput, sOutput, layers, fnActivation=nn.ReLU
        ).to(device)

        model.load_state_dict(torch.load(f"models/model{i}.pth"))

        model_list.append(model)

    accuracy = evaluate_ensemble(model_list, test_dataloader, device)

    with open("ensemble_accuracy.json", "w") as f:
        json.dump(accuracy, f)


if __name__ == "__main__":
    main()
