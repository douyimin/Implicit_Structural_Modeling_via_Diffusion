"""NPZ data contract and sparse horizon sampling for the release workflow."""

import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize(array):
    array = np.asarray(array, dtype=np.float32)
    return (array - array.min()) / (np.ptp(array) + 1e-6)


def load_arrays(path, keys):
    with np.load(path, allow_pickle=False) as archive:
        missing = set(keys) - set(archive.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        arrays = {key: np.asarray(archive[key], dtype=np.float32) for key in keys}
    shape = arrays[keys[0]].shape
    if len(shape) != 2 or min(shape) < 64 or any(n % 64 for n in shape):
        raise ValueError(f"{path}: expected 2D arrays with dimensions divisible by 64; got {shape}")
    for key, value in arrays.items():
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{path}: {key} must be finite and have shape {shape}")
    return arrays


def load_conditions(path):
    arrays = load_arrays(path, ("horiz", "fault"))
    for key, value in arrays.items():
        if value.min() < 0 or value.max() > 1:
            raise ValueError(f"{path}: {key} must be in [0, 1]")
    arrays["fault"] = (arrays["fault"] > 0.5).astype(np.float32)
    return arrays


def sample_horizons(rgt, count, rng, width=0.01):
    """Retain the original value-band contour extraction (zero = unknown)."""
    horizon = np.zeros_like(rgt)
    for _ in range(count):
        start = rng.uniform(0, 1 - width)
        mask = (rgt > start) & (rgt <= start + width)
        if mask.any():
            horizon[mask] = rgt[mask].mean()
    return horizon


def read_split(path):
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = [split[key] for key in ("train", "val", "test")]
    all_files = [name for group in groups for name in group]
    if any(not group for group in groups) or len(set(all_files)) != len(all_files):
        raise ValueError("Split must contain nonempty, disjoint train/val/test lists")
    for name in all_files:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Split entries must be relative paths within the data directory")
    return split


class StructuralDataset(Dataset):
    def __init__(self, root, files, training=False, seed=2026, horizon_min=5, horizon_max=15):
        self.root = Path(root).resolve()
        self.files = list(files)
        self.training, self.seed = training, seed
        self.horizon_min, self.horizon_max = horizon_min, horizon_max
        if not 1 <= horizon_min <= horizon_max:
            raise ValueError("Require 1 <= horizon_min <= horizon_max")
        for name in self.files:
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f"Missing or out-of-root sample: {name}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        arrays = load_arrays(self.root / self.files[index], ("rgt", "fault"))
        if arrays["rgt"].shape != (512, 512):
            raise ValueError("Training samples must be 512 x 512; no implicit resizing is applied")
        rgt = normalize(arrays["rgt"])
        fault = (arrays["fault"] > 0.5).astype(np.float32)
        rng = random if self.training else random.Random(self.seed + index)
        horiz = sample_horizons(rgt, rng.randint(self.horizon_min, self.horizon_max), rng)
        return {
            "jpg": torch.from_numpy(np.repeat(rgt[None], 3, axis=0)) * 2 - 1,
            "fault": torch.from_numpy(fault[None]) * 2 - 1,
            "horiz": torch.from_numpy(horiz[None]) * 2 - 1,
        }
