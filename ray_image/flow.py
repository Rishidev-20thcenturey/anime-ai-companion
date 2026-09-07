import torch


def sample_flow_pair(x0: torch.Tensor):
    """Create a linear noise/data interpolation for flow-matching training.

    x0 is a clean latent. t=0 is clean data and t=1 is Gaussian noise.
    Returns x_t, t and the target velocity (noise - data).
    """
    noise = torch.randn_like(x0)
    t = torch.rand(x0.shape[0], device=x0.device)
    shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)
    xt = (1 - t.view(shape)) * x0 + t.view(shape) * noise
    target = noise - x0
    return xt, t, target


@torch.no_grad()
def euler_sample(model, text, shape, steps=20, device=None, text_mask=None):
    """Generate a latent by integrating the learned velocity field."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    device = device or text.device
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((shape[0],), 1.0 - i * dt, device=device)
        # Training uses data->noise; sampling follows noise->data.
        v = model(x, t, text, text_mask=text_mask)
        x = x - dt * v
    return x
