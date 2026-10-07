"""Sample RGT fields from sparse faults and horizons with deterministic DDIM."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import yaml

from ISM_diffusion.weights import add_download_arguments, resolve_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    p.add_argument("--conditions", type=Path, default=Path("examples/flower_conditions.npz"))
    p.add_argument("--checkpoint", type=Path,
                   help="Local weights; omit to download ISM_PretrainedModel.ckpt from ModelScope")
    add_download_arguments(p)
    p.add_argument("--output", type=Path, default=Path("outputs/flower"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=1, help="Uses consecutive seeds starting at --seed")
    p.add_argument("--device", default="cuda", help="cpu, cuda, or cuda:N")
    p.add_argument("--steps", type=int)
    p.add_argument("--fault-scale", type=float)
    p.add_argument("--horizon-scale", type=float)
    p.add_argument("--ema", action="store_true", help="Use EMA stored by this release's training entry point")
    return p.parse_args()


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from pytorch_lightning import seed_everything
    from ISM_diffusion.data import load_conditions, normalize
    from ISM_diffusion.model import create_model, load_state_dict
    from ISM_diffusion.models.diffusion.ddim import DDIMSampler
    from ISM_diffusion.training import checkpoint_weights

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sampling = config["sampling"]
    for argument, key in (("steps", "ddim_steps"), ("fault_scale", "fault_scale"), ("horizon_scale", "horizon_scale")):
        if getattr(args, argument) is not None:
            sampling[key] = getattr(args, argument)
    if args.num_samples < 1 or not 0 <= args.seed < 2**32 or args.seed + args.num_samples > 2**32:
        raise ValueError("Sample count must be positive and all seeds must fit in uint32")
    # The inherited uniform DDIM schedule adds 1 to time indices.
    steps = sampling["ddim_steps"]
    if not 1 <= steps < 1000 or 1000 % steps:
        raise ValueError("DDIM steps must divide 1000 and be smaller than 1000 (e.g. 20, 50, 100)")
    if sampling["eta"] != 0 or not all(np.isfinite(sampling[k]) and sampling[k] >= 0 for k in ("fault_scale", "horizon_scale")):
        raise ValueError("Require eta=0 and finite, nonnegative control scales")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new --output")
    arrays = load_conditions(args.conditions)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install a matching PyTorch build or use --device cpu")
    args.checkpoint = resolve_checkpoint(args.checkpoint, cache_dir=args.cache_dir, revision=args.model_revision)
    seed_everything(args.seed)
    model = create_model(config["model"])
    if args.ema:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint_weights(checkpoint, use_ema=True), strict=True)
        del checkpoint
    else:
        model.load_state_dict(load_state_dict(args.checkpoint), strict=True)
    model = model.to(device).eval().requires_grad_(False)
    model.control_fault_scales = [sampling["fault_scale"]] * 13
    model.control_horiz_scales = [sampling["horizon_scale"]] * 13
    conditions = {key: torch.from_numpy(value)[None, None].to(device) * 2 - 1 for key, value in arrays.items()}
    conditions["c_crossattn"] = [model.clip_txt]
    height, width = arrays["horiz"].shape
    sampler = DDIMSampler(model, batch_classifier_free_guidance=False)
    args.output.mkdir(parents=True, exist_ok=True)
    record = dict(config=config, seeds=list(range(args.seed, args.seed + args.num_samples)),
                  checkpoint=str(args.checkpoint), conditions=str(args.conditions), ema=args.ema,
                  model_revision=args.model_revision,
                  device=str(device), torch=torch.__version__, input_shape=[height, width])
    (args.output / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    for seed in record["seeds"]:
        seed_everything(seed)
        autocast = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with torch.no_grad(), autocast:
            latent, _ = sampler.sample(steps, 1, (4, height // 8, width // 8), conditions,
                                       verbose=False, eta=0.0, unconditional_guidance_scale=1.0)
            decoded = model.decode_first_stage(latent)[0].mean(dim=0).float().cpu().numpy()
        rgt = normalize((np.clip(decoded, -1, 1) + 1) / 2)
        if not np.isfinite(rgt).all():
            raise RuntimeError("Generated RGT contains nonfinite values")
        stem = args.output / f"sample_{seed:010d}"
        np.savez_compressed(stem.with_suffix(".npz"), rgt=rgt, **arrays, seed=np.uint32(seed))
        colors = plt.get_cmap("jet")(rgt)
        layers = plt.get_cmap("tab20")(rgt)
        layers = np.where((arrays["horiz"] > 0)[..., None], plt.get_cmap("jet")(arrays["horiz"]), layers)
        layers[arrays["fault"] > 0.5] = [1, 1, 1, 1]
        plt.imsave(stem.with_suffix(".png"), np.concatenate([colors, layers], axis=0))
        print(f"Saved {stem}.npz and .png")


if __name__ == "__main__":
    main()
