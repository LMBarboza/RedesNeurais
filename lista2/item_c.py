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
    learning_rate = config.getfloat("TRAINING", "learning_rate")
    epochs = config.getint("TRAINING", "epochs")
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

    hiddenLayers = [128, 64]
    model = RedeFactory.createRede(
        sInput, sOutput, hiddenLayers, fnActivation=nn.ReLU
    ).to(device)

    print(model)

    optimizer1 = optim.Adam(model.parameters(), lr=2)
    optimizer2 = optim.Adam(model.parameters(), lr=1)
    optimizer3 = optim.Adam(model.parameters(), lr=0.1)
    optimizer4 = optim.Adam(model.parameters(), lr=0.01)

    loss = nn.CrossEntropyLoss()

    strategy1 = STDStrategy(loss, optimizer1)
    strategy2 = STDStrategy(loss, optimizer2)
    strategy3 = STDStrategy(loss, optimizer3)
    strategy4 = STDStrategy(loss, optimizer4)

    trainer1 = Trainer(model, strategy1)
    trainer2 = Trainer(model, strategy2)
    trainer3 = Trainer(model, strategy3)
    trainer4 = Trainer(model, strategy4)

    accuracy_list1 = trainer1.train(train_dataloader, test_dataloader, epochs, device)
    accuracy_list2 = trainer2.train(train_dataloader, test_dataloader, epochs, device)
    accuracy_list3 = trainer3.train(train_dataloader, test_dataloader, epochs, device)
    accuracy_list4 = trainer4.train(train_dataloader, test_dataloader, epochs, device)

    with open("accuracy.json", "w") as f:
        json.dump(accuracy_list1, f)
        json.dump(accuracy_list2, f)
        json.dump(accuracy_list3, f)
        json.dump(accuracy_list4, f)


if __name__ == "__main__":
    main()
