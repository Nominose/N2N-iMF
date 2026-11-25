"""Denoise an axial CT volume with a trained N2N-iMF generator."""

import argparse
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from n2nimf.backbone import ConditionalUNet
from n2nimf.checkpoint import load, load_generator
from n2nimf.data import PRESETS, IntensityTransform, conditions, load_volumes
from n2nimf.flow import sample


@torch.inference_mode()
def denoise(model, volumes, transform, nfe, k, device):
    height, width, depth = volumes[0].shape
    output = np.empty((height, width, depth), dtype=np.float32)
    for z in range(depth):
        inputs = conditions(volumes, z)
        total = np.zeros((height, width), dtype=np.float64)
        for condition in inputs:
            encoded = (
                torch.from_numpy(transform.encode(condition)).unsqueeze(0).to(device)
            )
            encoded = F.pad(encoded, (0, -width % 8, 0, -height % 8), mode="replicate")
            for _ in range(k):
                draw = sample(model, encoded, nfe)[0, 0, :height, :width].cpu().numpy()
                total += transform.decode(draw)
        output[:, :, z] = total / (k * len(inputs))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=PRESETS)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--volume", type=Path, help="brain CT or PCCT volume in HU")
    parser.add_argument(
        "--odd", type=Path, help="abdominal CT odd projection reconstruction"
    )
    parser.add_argument(
        "--even", type=Path, help="abdominal CT even projection reconstruction"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--nfe", type=int, help="network evaluations per generated output"
    )
    parser.add_argument(
        "--k", type=int, help="independent draws per conditioning input"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    preset = PRESETS[args.dataset]
    k = args.k if args.k is not None else preset["k"]
    nfe = args.nfe if args.nfe is not None else preset["nfe"]
    if min(k, nfe) < 1:
        parser.error("K and NFE must be positive.")
    if args.dataset == "abdominal":
        if not args.odd or not args.even or args.volume:
            parser.error("For abdominal CT, supply --odd and --even.")
        paths = [args.odd, args.even]
        if args.odd.resolve() == args.even.resolve():
            parser.error("--odd and --even must be different reconstructions.")
    else:
        if not args.volume or args.odd or args.even:
            parser.error("For brain CT or PCCT, supply --volume.")
        paths = [args.volume]
    if args.output.exists() or not args.output.name.endswith((".nii", ".nii.gz")):
        parser.error("--output must be a new .nii or .nii.gz file.")
    ref, volumes = load_volumes(paths)
    if len(volumes) == 1 and ref.shape[-1] < 2:
        parser.error("Adjacent-slice conditioning requires at least two slices.")
    model = ConditionalUNet(preset["channels"])
    load_generator(model, load(args.checkpoint), args.dataset, args.weights)
    model.to(args.device).eval()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    output = denoise(
        model, volumes, IntensityTransform(args.dataset), nfe, k, args.device
    )
    if not np.isfinite(output).all():
        raise ValueError("Prediction contains nonfinite intensities.")
    header = ref.header.copy()
    header.set_data_dtype(np.float32)
    result = ref.__class__(output, ref.affine, header)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".nii.gz" if args.output.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(
        dir=args.output.parent, suffix=suffix, delete=False
    ) as f:
        tmp = Path(f.name)
    try:
        nib.save(result, str(tmp))
        os.replace(tmp, args.output)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"Wrote {args.output} (NFE={nfe}, K={k} per input)")


if __name__ == "__main__":
    main()
