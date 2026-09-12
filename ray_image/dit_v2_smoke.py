import argparse

import torch

from .dit_v2 import RAYDiT_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test RAYDiT_v2 forward pass")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu; defaults to CUDA when available")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = RAYDiT_v2(
        latent_channels=16,
        latent_size=32,
        dim=768,
        depth=18,
        heads=12,
        patch=2,
        cond_dim=2560,
        cross_gate=True,
    ).to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}")
    print(f"parameters: {params:,} ({params / 1e6:.2f}M)")

    latents = torch.randn(1, 16, 32, 32, device=device)
    text = torch.randn(1, 47, 2560, device=device)
    timestep = torch.tensor([500], device=device, dtype=torch.long)

    with torch.inference_mode():
        output = model(latents, timestep, text)

    expected = (1, 16, 32, 32)
    assert tuple(output.shape) == expected, (
        f"unexpected output shape: {tuple(output.shape)} != {expected}"
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"peak VRAM: {peak:.3f} GiB")
    else:
        print("peak VRAM: N/A (CPU run)")

    print(f"output shape: {tuple(output.shape)}")
    print("PASS: DiT v2 forward shape verified")


if __name__ == "__main__":
    main()
