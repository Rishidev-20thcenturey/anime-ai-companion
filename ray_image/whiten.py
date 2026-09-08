"""Deterministic channel-wise latent whitening for RAY-IMAGE (N2).

Hypothesis under test (single change): the flow-matching generator struggles to
learn shape because the raw VAE latents are strongly channel-imbalanced (the N1
statistics showed per-channel means/std far from zero/one). Normalizing each
latent channel to ~zero mean / unit variance rebalances the flow objective
across channels and brings the data prior closer to the standard-Gaussian noise
used by ``flow.sample_flow_pair`` / ``euler_sample``.

This is a data-level pre/post-processing only. It does NOT change the VAE, text
encoder, DiT, tokenizer, flow objective, sampler, dataset, or evaluator.

Convention (kept identical in training and generation):
    z_norm = (z - mean_c) / std_c          per latent channel c
    z      = z_norm * std_c + mean_c       (inverse, used before VAE decode)

The mean/std come from the N1 probe output
(``vae_latent_stats.json`` -> ``per_channel_mean`` / ``per_channel_std``),
computed with the SAME VAE checkpoint used for generator training.
"""
import json

import torch


class LatentNormalizer:
    """Stores per-channel mean/std and applies forward/inverse whitening."""

    def __init__(self, mean, std, eps=1e-6):
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        if mean.ndim != 1 or std.ndim != 1 or mean.numel() != std.numel():
            raise ValueError("mean and std must be 1-D with equal length")
        self.mean = mean
        self.std = std.clamp_min(eps)

    @classmethod
    def from_stats_dict(cls, stats):
        """Build from a probe stats dict (per_channel_mean / per_channel_std)."""
        return cls(stats["per_channel_mean"], stats["per_channel_std"])

    @classmethod
    def from_stats_json(cls, path):
        with open(path) as f:
            return cls.from_stats_dict(json.load(f))

    @classmethod
    def from_state(cls, state):
        return cls(state["mean"], state["std"])

    @property
    def channels(self):
        return int(self.mean.numel())

    def _bc(self, z):
        # Broadcast [C] stats to z's [B,C,H,W] layout.
        return self.mean.to(z.device).view(1, -1, 1, 1), \
               self.std.to(z.device).view(1, -1, 1, 1)

    def normalize(self, z):
        """(z - mean_c) / std_c per channel."""
        mean, std = self._bc(z)
        return (z - mean) / std

    def denormalize(self, z_norm):
        """z_norm * std_c + mean_c per channel (inverse of normalize)."""
        mean, std = self._bc(z_norm)
        return z_norm * std + mean

    def state(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    def validate_channels(self, channels):
        if self.channels != channels:
            raise ValueError(
                f"whitening stats have {self.channels} channels but model "
                f"latent has {channels}"
            )
