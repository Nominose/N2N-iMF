"""Training objective and sampling rule for Noise2Noise with Improved Mean Flow."""

import itertools

import torch


def weighted_mse(prediction, target):
    error = (prediction - target).square().mean(dim=(1, 2, 3))
    return (error / (error.detach() + 0.01)).mean()


def training_loss(model, condition, target):
    batch = target.shape[0]
    noise = torch.randn_like(target)
    draws = (torch.randn(2, batch, device=target.device) - 0.4).sigmoid()
    end = draws.amax(dim=0).clamp(1e-4, 1 - 1e-4)
    start = torch.minimum(draws.amin(dim=0), end - 1e-4)
    start = torch.where(torch.rand_like(end) > 0.5, end, start)
    t = end[:, None, None, None]
    x = (1 - t) * target + t * noise
    velocity = noise - target
    with torch.no_grad():
        tangent, _ = model(x, condition, end, end)

    # Move along the predicted trajectory while keeping the lower endpoint fixed.
    def field(state, time):
        return model(state, condition, start, time)

    average, derivative, auxiliary = torch.func.jvp(
        field, (x, end), (tangent, torch.ones_like(end)), has_aux=True
    )
    instantaneous = average + (end - start)[:, None, None, None] * derivative.detach()
    return weighted_mse(instantaneous, velocity) + 0.5 * weighted_mse(
        auxiliary, velocity
    )


def sample(model, condition, nfe):
    """Generate one independent target per input, in model intensity units."""
    x = torch.randn_like(condition[:, :1])
    times = torch.linspace(1, 0, nfe + 1, device=x.device, dtype=x.dtype)
    for end, start in itertools.pairwise(times):
        u, _ = model(x, condition, start.expand(len(x)), end.expand(len(x)))
        x = x - (end - start) * u
    return x
