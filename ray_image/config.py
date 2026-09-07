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
        return self.image_size // 8
