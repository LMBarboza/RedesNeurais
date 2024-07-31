import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, latent_size: int, mlp: bool = False) -> None:
        super().__init__()
        self.encoder = nn.Sequential()
        self.classifier = nn.Linear(latent_size, 10)
        self.decoder = nn.Sequential()
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_encode = self.encoder(x)
        x_decode = self.decoder(x_encode)
        return self.classifier(x_encode), x_decode


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for module in self.encoder[:-2]:
            x = module(x)

        mu = self.encoder[-1](x)
        logvar = self.encoder[-2](x)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        return self.decode(z), mu, logvar
