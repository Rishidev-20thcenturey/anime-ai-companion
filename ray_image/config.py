from dataclasses import dataclass


@dataclass
class RAYConfig:
    image_size: int = 64
    latent_channels: int = 4
    vae_base: int = 32
    model_dim: int = 256
    depth: int = 6
    heads: int = 4
    patch_size: int = 2
    text_dim: int = 256
    vocab_size: int = 8192
    max_tokens: int = 64

    @property
    def latent_size(self) -> int:
        """Spatial latent size for the VAE's fixed 8x compression."""
        if self.image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8")
        return self.image_size // 8

    @property
    def latent_tokens(self) -> int:
        """Number of DiT patch tokens produced by the latent grid."""
        if self.latent_size % self.patch_size != 0:
            raise ValueError("latent_size must be divisible by patch_size")
        side = self.latent_size // self.patch_size
        return side * side
