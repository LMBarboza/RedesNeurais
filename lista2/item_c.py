import argparse
import json
import configparser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modules.rede_factory import RedeFactory
from modules.trainer import Trainer
from modules.treino_strategy import STDStrategy


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
    epochs = config.getint("TRAINING", "epochs")
    learning_rate = config.getfloat("TRAINING", "learning_rate")
    use_cuda = config.getboolean("TRAINING", "cuda")

    cuda = use_cuda and torch.cuda.is_available()

    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    sInput = 28 * 28
    sOutput = 10

    hiddenLayers = [128]
    accuracy_list = []

    loss = nn.CrossEntropyLoss()
    print(f"Learning Rate: {learning_rate}")

    model = RedeFactory.createRede(
        sInput, sOutput, hiddenLayers, fnActivation=nn.ReLU
    ).to(device)

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    strategy = STDStrategy(loss, optimizer)

    trainer = Trainer(model, strategy)

    accuracy = trainer.train(train_dataloader, test_dataloader, epochs, device)

    accuracy_list.append(accuracy)

    with open("lr_accuracy.json", "w") as f:
        json.dump(accuracy_list, f)


if __name__ == "__main__":
    main()
