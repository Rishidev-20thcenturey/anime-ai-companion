import torch
from torch import nn


class RAYVAE(nn.Module):
    """Tiny 8x-compression VAE prototype: 512x512 -> 64x64x4."""

    def __init__(self, latent_channels: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(256, latent_channels * 2, 3, 1, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, 3, 1, 1), nn.SiLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        stats = self.encoder(x)
        mean, logvar = stats.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar.clamp(-30, 20))
        z = mean + std * torch.randn_like(std)
        return z, mean, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        return self.decode(z), z, mean, logvar
