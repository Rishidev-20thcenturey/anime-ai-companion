import torch
from torch import nn


class RAYVAE(nn.Module):
    """Compact 8x spatial-compression VAE prototype.

    For the default 64x64 prototype this maps:
        RGB image:  [B, 3, 64, 64]
        latent:     [B, 16, 8, 8]

    The architecture is intentionally small for free-GPU experimentation and
    can later be widened/deepened for the larger RAY-IMAGE configurations.
    """

    def __init__(self, latent_channels: int = 16, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(c, c * 2, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(c * 4, latent_channels * 2, 3, 1, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, c * 4, 3, 1, 1), nn.SiLU(),
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(c, 3, 4, 2, 1), nn.Sigmoid(),
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
