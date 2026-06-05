#!/usr/bin/env python
"""
Demo: implicit structural modeling via a conditional diffusion framework.

Given a sparse horizon field and a fault skeleton, this script reconstructs a
dense relative-geological-time (RGT) scalar field with the trained model and
saves a visualization. It is a minimal, self-contained inference example for the
manuscript "Implicit Structural Modeling via Generative Diffusion Framework".

Example
-------
    python demo.py \
        --cond      val_data/A_SSZ_horiz.npz \
        --ckpt      512x512.ckpt \
        --config    ./models/cldm_v21.yaml \
        --clip-txt  clip_txt.npy \
        --strength  1.5 \
        --steps     50 \
        --seed      0 \
        --out       output/demo.png

Inputs
------
--cond     A .npz file holding two arrays:
             'horiz' : sparse horizon field, normalised to [0, 1]
                       (0 marks unconstrained pixels; >0 are RGT iso-levels)
             'fault' : fault skeleton; binarised at 0.5 inside the script
--ckpt     Model weights (the file name is arbitrary).
--clip-txt A fixed text-embedding tensor saved as .npy. It is text-independent:
           the model does not use a textual prompt, and this fixed embedding is
           passed only to satisfy the cross-attention interface of the backbone.

Notes
-----
- Inference uses DDIM with eta=0 (deterministic). The conditioning strength
  --strength sets the per-branch injection scale s_k for both the fault and the
  horizon branch (s=1.5 is the default reported in the paper).
- A CUDA GPU is recommended. Pass --device cpu to run on CPU (much slower).
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything

from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


# Number of conditional injection points in the backbone (UNet encoder + middle).
N_INJECTION = 13
# Working resolution fed to the model (height, width).
INPUT_HW = (512, 1280)
# VAE spatial compression factor.
VAE_DOWNSAMPLE = 8


def normalization(data):
    """Min-max normalise an array to [0, 1]."""
    rng = np.max(data) - np.min(data)
    return (data - np.min(data)) / (rng + 1e-6)


def parse_args():
    p = argparse.ArgumentParser(
        description="Implicit structural modeling demo (conditional diffusion).")
    p.add_argument('--cond', required=True,
                   help="Conditional input .npz with 'horiz' and 'fault' arrays.")
    p.add_argument('--ckpt', default='512x512.ckpt',
                   help='Path to the model weights.')
    p.add_argument('--config', default='./models/cldm_v21.yaml',
                   help='Path to the model config YAML.')
    p.add_argument('--clip-txt', default='clip_txt.npy',
                   help='Path to the fixed (text-independent) embedding .npy.')
    p.add_argument('--strength', type=float, default=1.5,
                   help='Conditioning injection strength s (default: 1.5).')
    p.add_argument('--steps', type=int, default=50,
                   help='Number of DDIM sampling steps (default: 50).')
    p.add_argument('--seed', type=int, default=0,
                   help='Random seed (default: 0).')
    p.add_argument('--out', default='output/demo.png',
                   help='Output image path.')
    p.add_argument('--device', default='cuda',
                   choices=['cuda', 'cpu'], help='Compute device.')
    return p.parse_args()


def load_conditions(cond_path, device):
    """Load and preprocess the horizon and fault conditions."""
    cond = np.load(cond_path)
    if 'horiz' not in cond or 'fault' not in cond:
        raise KeyError(
            f"{cond_path} must contain 'horiz' and 'fault' arrays; "
            f"found {list(cond.keys())}.")

    # Horizons: continuous RGT iso-values mapped from [0, 1] to [-1, 1].
    horiz = torch.from_numpy(cond['horiz']).float().to(device)[None, None] * 2 - 1
    # Faults: binarised at 0.5, then mapped to {-1, 1}.
    fault = torch.from_numpy((cond['fault'] > 0.5)).float().to(device)[None, None] * 2 - 1

    # Resize to the model's working resolution.
    horiz = F.interpolate(horiz, INPUT_HW)
    fault = F.interpolate(fault, INPUT_HW)
    return horiz, fault


def build_model(config, ckpt, device):
    """Instantiate the model and load weights."""
    model = create_model(config).cpu()
    model.load_state_dict(load_state_dict(ckpt, location='cpu'), strict=False)
    return model.to(device)


def run(args):
    seed_everything(args.seed)
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    if device != args.device:
        print('[demo] CUDA not available; falling back to CPU.')

    # Fixed, text-independent embedding for the cross-attention interface.
    clip_txt = torch.from_numpy(np.load(args.clip_txt)).float().to(device)

    horiz, fault = load_conditions(args.cond, device)

    model = build_model(args.config, args.ckpt, device)
    ddim_sampler = DDIMSampler(model)

    # Per-branch injection strength s_k (paper default: 1.5). The seismic branch,
    # if present in the backbone, is disabled in this demo.
    model.control_fault_scales = [args.strength] * N_INJECTION
    model.control_horiz_scales = [args.strength] * N_INJECTION
    model.control_seis_scales = [0.0] * N_INJECTION

    B, C, H, W = horiz.shape
    latent_shape = (4, H // VAE_DOWNSAMPLE, W // VAE_DOWNSAMPLE)
    cond = {'c_crossattn': [clip_txt], 'fault': fault, 'horiz': horiz}

    autocast = (torch.autocast(device_type='cuda')
                if device == 'cuda' else torch.no_grad())
    with torch.no_grad():
        with autocast:
            samples, _ = ddim_sampler.sample(
                args.steps, B, latent_shape, cond,
                verbose=False, eta=0.0, unconditional_guidance_scale=1.0)
            x_samples = model.decode_first_stage(samples)

    # Decode to a single-channel RGT field in [0, 1].
    rgt = np.clip(x_samples[0].mean(dim=0).float().cpu().numpy(), -1, 1)
    rgt = normalization((rgt + 1) / 2)

    # Compose a two-panel visualization:
    #   top    -- the reconstructed RGT field (jet colormap)
    #   bottom -- the same field with the input horizon constraints overlaid
    horiz_show = (horiz.float().cpu().numpy()[0, 0] + 1) / 2
    constraint_mask = (horiz_show > 0.0).astype(np.float32)[..., None]
    horiz_overlay = plt.get_cmap('jet')(horiz_show)

    rgt_jet = plt.get_cmap('jet')(rgt)
    rgt_tab = plt.get_cmap('tab20')(rgt)
    rgt_tab = np.where(constraint_mask, horiz_overlay, rgt_tab)
    panel = np.concatenate([rgt_jet, rgt_tab], axis=0)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.imsave(args.out, panel)
    print(f"[demo] saved result to {args.out} "
          f"(seed={args.seed}, strength={args.strength}, steps={args.steps})")


if __name__ == '__main__':
    run(parse_args())
