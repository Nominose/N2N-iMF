"""Noisy CT pairs, intensity transforms and sampling settings."""

import csv
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import IterableDataset, get_worker_info

# Intensity limits are the model input ranges in HU.
PRESETS = {
    "brain": {
        "channels": 2,
        "hu": (-1000, 2000),
        "patch": 128,
        "slices": 50,
        "batch": 2,
        "nfe": 3,
        "k": 10,
    },
    "abdominal": {
        "channels": 1,
        "hu": (-200, 250),
        "patch": 256,
        "slices": 50,
        "batch": 1,
        "nfe": 3,
        "k": 10,
    },
    "pcct": {
        "channels": 2,
        "hu": (-1000, 1000),
        "patch": 128,
        "slices": 48,
        "batch": 32,
        "nfe": 5,
        "k": 20,
    },
}


class IntensityTransform:
    def __init__(self, dataset):
        self.low, self.high = PRESETS[dataset]["hu"]
        self.bins = None
        if dataset == "brain":
            assets = Path(__file__).with_name("assets")
            self.bins = (
                np.load(assets / "bins.npy"),
                np.load(assets / "bins_mapped.npy"),
            )

    def transfer(self, x, reverse=False):
        if self.bins is None:
            return x
        source, dest = self.bins[::-1] if reverse else self.bins
        mask = (x > source[0]) & (x < source[-1])
        return np.where(mask, np.interp(x, source, dest), x).astype(np.float32)

    def encode(self, x):
        mapped = self.transfer(x)
        return (np.clip(mapped, self.low, self.high) - self.low) * (
            2 / (self.high - self.low)
        ) - 1

    def decode(self, x):
        hu = (x + 1) * ((self.high - self.low) / 2) + self.low
        return self.transfer(hu, reverse=True)


def load_volumes(paths):
    images = [nib.load(str(path)) for path in paths]
    ref = images[0]
    if len(ref.shape) != 3 or min(ref.shape) == 0:
        raise ValueError("Expected a nonempty 3D axial volume.")
    for image in images[1:]:
        if image.shape != ref.shape or not np.allclose(
            image.affine, ref.affine, atol=1e-4, rtol=0
        ):
            raise ValueError(
                "The two split reconstructions must have the same shape and affine."
            )
    volumes = [image.get_fdata(dtype=np.float32) for image in images]
    if not all(np.isfinite(x).all() for x in volumes):
        raise ValueError("Volumes must contain finite HU values.")
    return ref, volumes


def conditions(volumes, z):
    if len(volumes) == 2:
        return [vol[:, :, z][None] for vol in volumes]
    vol = volumes[0]
    previous, following = max(0, z - 1), min(vol.shape[2] - 1, z + 1)
    return [np.stack((vol[:, :, previous], vol[:, :, following]))]


class TrainingPairs(IterableDataset):
    def __init__(self, manifest, dataset, patch):
        super().__init__()
        keys = ("odd", "even") if dataset == "abdominal" else ("volume",)
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not set(keys).issubset(reader.fieldnames or ()):
                raise ValueError(f"Expected manifest columns {keys}.")
            self.cases = [
                tuple(manifest.parent / row[key] for key in keys) for row in reader
            ]
        if not self.cases:
            raise ValueError("Training manifest is empty.")
        self.transform = IntensityTransform(dataset)
        self.slices = PRESETS[dataset]["slices"]
        self.patch = patch

    def __iter__(self):
        worker = get_worker_info()
        cases = (
            self.cases
            if worker is None
            else self.cases[worker.id :: worker.num_workers]
        )
        for paths in random.sample(cases, len(cases)):
            ref, volumes = load_volumes(paths)
            h, w, depth = ref.shape
            if min(h, w) < self.patch or (len(volumes) == 1 and depth < 3):
                raise ValueError(
                    f"Volume {paths[0]} is too small for the selected training patch or slice pair."
                )
            volumes = [self.transform.encode(vol) for vol in volumes]
            eligible = range(depth) if len(volumes) == 2 else range(1, depth - 1)
            for z in random.sample(eligible, min(self.slices, len(eligible))):
                for _ in range(2):
                    x = random.randint(0, h - self.patch)
                    y = random.randint(0, w - self.patch)
                    cond = conditions(volumes, z)
                    if len(cond) == 2:
                        if random.random() < 0.5:
                            cond.reverse()
                        image, target = cond
                    else:
                        image, target = cond[0], volumes[0][:, :, z][None]
                    pair = np.concatenate((image, target))[
                        :, x : x + self.patch, y : y + self.patch
                    ]
                    if random.random() < 0.5:
                        angle = random.uniform(-10, 10)
                        shift = (random.randint(-10, 10), random.randint(-10, 10))
                        for channel in pair:
                            fill = float(channel.min())
                            rotated = ndimage.rotate(
                                channel, angle, reshape=False, order=1, cval=fill
                            )
                            channel[:] = ndimage.shift(
                                rotated, shift, order=0, cval=fill
                            )
                    pair = torch.from_numpy(np.ascontiguousarray(pair))
                    yield pair[:-1], pair[-1:]
