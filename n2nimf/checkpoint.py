"""Load release checkpoints and matching two-head research checkpoints."""

import torch


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def load_generator(model, checkpoint, dataset, weights):
    if checkpoint.get("dataset", dataset) != dataset:
        raise ValueError("This checkpoint was trained for a different dataset.")
    if weights == "ema":
        state = {
            k.removeprefix("ema_model."): v
            for k, v in checkpoint["ema"].items()
            if k.startswith("ema_model.")
        }
    else:
        state = checkpoint["model"]
    prefix = "wrapped_model.base_unet."
    model.load_state_dict(
        {k.removeprefix(prefix): v for k, v in state.items()}, strict=True
    )
