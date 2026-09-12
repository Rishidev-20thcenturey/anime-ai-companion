"""Smoke test for the N8 256x256 RAYVAE_v2 architecture."""

import torch

from .vae_v2 import RAYVAE_v2


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("vae_v2_smoke requires CUDA so VRAM usage can be checked")

    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = RAYVAE_v2(latent_channels=16, base_channels=64).to(device).eval()
    x = torch.rand(1, 3, 256, 256, device=device)

    with torch.inference_mode():
        z, mean, logvar = model.encode(x)
        assert tuple(z.shape) == (1, 16, 32, 32), z.shape
        assert tuple(mean.shape) == (1, 16, 32, 32), mean.shape
        assert tuple(logvar.shape) == (1, 16, 32, 32), logvar.shape
        recon = model.decode(z)
        assert tuple(recon.shape) == (1, 3, 256, 256), recon.shape

    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f"latent_shape={tuple(z.shape)}")
    print(f"recon_shape={tuple(recon.shape)}")
    print(f"peak_vram_gb={peak_gb:.3f}")
    assert peak_gb < 8.0, f"peak VRAM {peak_gb:.3f} GB exceeds 8 GB budget"
    print("vae_v2_smoke: PASS (<8 GB VRAM)")


if __name__ == "__main__":
    main()
