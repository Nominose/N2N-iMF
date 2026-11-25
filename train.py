"""Train N2N-iMF on paired noisy CT images."""

import argparse
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from ema_pytorch import EMA
from torch.utils.data import DataLoader

from n2nimf.backbone import ConditionalUNet
from n2nimf.checkpoint import load, load_generator
from n2nimf.data import PRESETS, TrainingPairs
from n2nimf.flow import sample, training_loss
from n2nimf.training import PatchDiscriminator, discriminator_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=PRESETS)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="checkpoint file, updated after each epoch",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--fine-tune-from", type=Path, help="initialize the high-pass PatchGAN stage"
    )
    group.add_argument(
        "--resume", action="store_true", help="continue training from --output"
    )
    parser.add_argument(
        "--init-weights",
        choices=("raw", "ema"),
        default="raw",
        help="initial generator for fine-tuning",
    )
    parser.add_argument(
        "--epochs", type=int, help="total epochs for this stage (base training: 200)"
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adv-weight", type=float, default=0.5)
    parser.add_argument(
        "--adv-nfe",
        type=int,
        default=1,
        help="sampling steps used to generate discriminator inputs",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = PRESETS[args.dataset]
    if args.output.exists() and not args.resume:
        parser.error("--output already exists. Choose another path or use --resume.")
    saved = load(args.output) if args.resume else None
    if saved and saved["dataset"] != args.dataset:
        parser.error("The saved checkpoint belongs to another dataset.")
    tune = saved["fine_tuning"] if saved else args.fine_tune_from is not None
    if tune and args.dataset == "abdominal":
        parser.error("The abdominal model uses the base training stage.")
    if args.epochs is None:
        if tune:
            parser.error("Specify the total --epochs for fine-tuning.")
        args.epochs = 200
    batch = args.batch_size if args.batch_size is not None else config["batch"]
    patch = args.patch_size if args.patch_size is not None else config["patch"]
    if (
        min(batch, args.epochs, args.adv_nfe) < 1
        or patch < (24 if tune else 8)
        or patch % 8
    ):
        parser.error(
            "Use positive counts and a patch divisible by 8 (at least 24 for fine-tuning)."
        )
    if args.lr <= 0 or args.workers < 0 or args.adv_weight < 0:
        parser.error("Invalid learning rate, worker count or adversarial weight.")
    start = saved["epoch"] if saved else 0
    step = saved["step"] if saved else 0
    if args.epochs <= start:
        parser.error("--epochs must exceed the completed epoch count.")
    torch.manual_seed(args.seed)
    model = ConditionalUNet(config["channels"]).to(args.device)
    if args.fine_tune_from:
        load_generator(
            model, load(args.fine_tune_from), args.dataset, args.init_weights
        )
    ema = EMA(
        model,
        beta=0.999 if tune else 0.995,
        update_every=10,
        include_online_model=False,
    ).to(args.device)
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.0, 0.99) if tune else (0.9, 0.95)
    )
    if tune:
        discriminator = PatchDiscriminator().to(args.device)
        brain = args.dataset == "brain"
        disc_opt = torch.optim.Adam(
            discriminator.parameters(), lr=4e-4 if brain else 2e-4, betas=(0.0, 0.99)
        )
        r1_gamma = 0.01 if brain else 1.0
    if saved:
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        opt.load_state_dict(saved["optimizer"])
        if tune:
            discriminator.load_state_dict(saved["discriminator"])
            disc_opt.load_state_dict(saved["disc_optimizer"])
            args.adv_weight = saved["adv_weight"]
            args.adv_nfe = saved["adv_nfe"]
        del saved
    train_set = TrainingPairs(args.manifest, args.dataset, patch)
    loader = DataLoader(train_set, batch_size=batch, num_workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, args.epochs):
        torch.manual_seed(args.seed + epoch)
        random.seed(args.seed + epoch)
        np.random.seed((args.seed + epoch) % (2**32))
        model.train()
        loss_sum, count = 0.0, 0
        for condition, target in loader:
            condition, target = condition.to(args.device), target.to(args.device)
            opt.zero_grad(set_to_none=True)
            loss = training_loss(model, condition, target)
            if tune:
                discriminator.requires_grad_(False)
                fake = sample(model, condition, args.adv_nfe)
                loss = loss - args.adv_weight * discriminator(fake).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            if tune:
                discriminator.requires_grad_(True)
                disc_opt.zero_grad(set_to_none=True)
                d_loss = discriminator_loss(discriminator, target, fake, step, r1_gamma)
                d_loss.backward()
                disc_opt.step()
            ema.update()
            step += 1
            loss_sum += loss.detach().item() * len(target)
            count += len(target)
        state = {
            "dataset": args.dataset,
            "epoch": epoch + 1,
            "step": step,
            "fine_tuning": tune,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": opt.state_dict(),
        }
        if tune:
            state.update(
                discriminator=discriminator.state_dict(),
                disc_optimizer=disc_opt.state_dict(),
                adv_weight=args.adv_weight,
                adv_nfe=args.adv_nfe,
            )
        with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as tmp:
            name = tmp.name
        try:
            torch.save(state, name)
            os.replace(name, args.output)
        finally:
            Path(name).unlink(missing_ok=True)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: {count} patches, mean generator loss {loss_sum / count:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
