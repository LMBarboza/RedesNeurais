import argparse
import configparser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modules.factory import FactoryProducer
from modules.trainer import Trainer
from modules.treino_strategy import (
    ClassificationStrategy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino Autoencoders")
    parser.add_argument(
        "--config",
        type=str,
        default="configurations.ini",
        help="PATH para configurações de treino",
    )

    parser.add_argument(
        "--latent_size",
        type=int,
        default=2,
        help="Dimensão espaço latente",
    )
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    latent_size = args.latent_size
    batch_size = config.getint("TRAINING", "batch_size")
    learning_rate = config.getfloat("TRAINING", "learning_rate")
    epochs = config.getint("TRAINING", "epochs")
    use_cuda = config.getboolean("TRAINING", "cuda")

    cuda = use_cuda and torch.cuda.is_available()

    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    conv_factory = FactoryProducer.getFactory("ConvAutoencoder")

    mlp_factory = FactoryProducer.getFactory("MLPAutoencoder")

    model_conv = conv_factory.createRede(
        sInput=1,
        sLatent=latent_size,
        sLayers=[16, 32, 64],
        fnActivation=nn.ReLU,
        sKernel=3,
        dropout=True,
    ).to(device)

    model_mlp = mlp_factory.createRede(
        sInput=784,
        sLatent=latent_size,
        sLayers=[128, 64, 32],
        fnActivation=nn.ReLU,
        sKernel=3,
        dropout=True,
    ).to(device)

    print(model_mlp)

    optimizer = optim.AdamW(model_mlp.parameters(), lr=learning_rate)

    classification_losses = [nn.CrossEntropyLoss(), nn.MSELoss()]

    classification_strategy = ClassificationStrategy(classification_losses, optimizer)

    trainer = Trainer(model_mlp, classification_strategy)
    accuracy = trainer.train(train_dataloader, test_dataloader, epochs, device)
    torch.save(model_mlp.state_dict(), f"models/model_mlp_classification_2.pth")

    print(model_conv)

    optimizer = optim.AdamW(model_conv.parameters(), lr=learning_rate)
    classification_strategy = ClassificationStrategy(classification_losses, optimizer)

    trainer = Trainer(model_conv, classification_strategy)
    accuracy = trainer.train(train_dataloader, test_dataloader, epochs, device)
    torch.save(model_conv.state_dict(), f"models/model_conv_classification_2.pth")


if __name__ == "__main__":
    main()
