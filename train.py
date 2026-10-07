"""Train the partially frozen model using the manuscript's confirmed settings."""
import argparse
import json
import os
from pathlib import Path

import yaml

from ISM_diffusion.weights import add_download_arguments, resolve_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    p.add_argument("--data", type=Path,
                   help="Local NPZ directory; omit to download and extract RGTSYN.7z from ModelScope")
    p.add_argument("--split", type=Path,
                   help="Local split manifest; omit to prepare a fixed-seed 3600/200/200 split")
    p.add_argument("--data-dir", type=Path, default=Path("data/modelscope"),
                   help="Directory for automatic data extraction and split manifests")
    checkpoint = p.add_mutually_exclusive_group()
    checkpoint.add_argument("--init", type=Path,
                            help="Local initialization weights; omit to download ISM_PretrainedModel.ckpt from ModelScope")
    checkpoint.add_argument("--resume", type=Path, help="Resume a full Lightning checkpoint from this entry point")
    add_download_arguments(p)
    p.add_argument("--output", type=Path, default=Path("runs/paper"))
    p.add_argument("--devices", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--workers", type=int)
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.strategies import DDPStrategy
    from torch.utils.data import DataLoader
    from ISM_diffusion.config import get_model_config
    from ISM_diffusion.data import StructuralDataset
    from ISM_diffusion.dataset_download import resolve_training_data, resolve_training_split
    from ISM_diffusion.model import load_state_dict
    from ISM_diffusion.training import PaperControlLDM, TrainableEMA

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = config["training"]
    for key in ("devices", "batch_size", "max_steps", "workers"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if min(settings[k] for k in ("devices", "batch_size", "max_steps", "save_every")) <= 0:
        raise ValueError("Devices, batch size, steps and checkpoint interval must be positive")
    if settings["workers"] < 0 or settings["accumulate_grad_batches"] != 1:
        raise ValueError("Workers must be nonnegative; this recipe uses no gradient accumulation")
    if not torch.cuda.is_available() or settings["devices"] > torch.cuda.device_count():
        raise RuntimeError("Not enough CUDA GPUs; use --devices for a smaller development run")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume and "LOCAL_RANK" not in os.environ:
        raise ValueError("Output directory is not empty; use a new path or --resume")
    if args.resume:
        args.resume = resolve_checkpoint(args.resume)
    args.data = resolve_training_data(args.data, storage_dir=args.data_dir,
                                      cache_dir=args.cache_dir, revision=args.model_revision)
    args.split, split = resolve_training_split(args.data, args.split, storage_dir=args.data_dir,
                                               seed=config["seed"], resume=args.resume)
    if not args.resume:
        args.init = resolve_checkpoint(args.init, cache_dir=args.cache_dir, revision=args.model_revision)
    pl.seed_everything(config["seed"], workers=True)
    model = PaperControlLDM(**get_model_config(config["model"])["params"])
    if args.init:
        model.load_state_dict(load_state_dict(args.init), strict=True)
    model.learning_rate = settings["learning_rate"]
    model.optimizer_options = {k: settings[k] for k in ("betas", "eps", "weight_decay")}
    loaders = []
    for subset in ("train", "val"):
        dataset = StructuralDataset(args.data, split[subset], training=subset == "train",
                                    seed=config["seed"], horizon_min=settings["horizon_min"],
                                    horizon_max=settings["horizon_max"])
        loaders.append(DataLoader(dataset, batch_size=settings["batch_size"],
                                  shuffle=subset == "train", num_workers=settings["workers"],
                                  pin_memory=True, persistent_workers=settings["workers"] > 0))
    checkpoint = ModelCheckpoint(dirpath=args.output / "checkpoints", filename="step-{step:08d}",
                                 every_n_train_steps=settings["save_every"], save_last=True,
                                 save_top_k=-1, save_on_train_epoch_end=False)
    ema = TrainableEMA(settings["ema_decay"])
    trainer = pl.Trainer(accelerator="gpu", devices=settings["devices"],
                         strategy=DDPStrategy(find_unused_parameters=False) if settings["devices"] > 1 else "auto",
                         precision=settings["precision"], max_steps=settings["max_steps"], max_epochs=-1,
                         accumulate_grad_batches=1, callbacks=[ema, checkpoint],
                         logger=CSVLogger(str(args.output), name="logs"),
                         default_root_dir=str(args.output), num_sanity_val_steps=0)
    if trainer.is_global_zero:
        args.output.mkdir(parents=True, exist_ok=True)
        record = dict(config=config, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      global_batch=settings["batch_size"] * settings["devices"],
                      torch=torch.__version__, lightning=pl.__version__)
        (args.output / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        (args.output / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    trainer.fit(model, loaders[0], loaders[1], ckpt_path=str(args.resume) if args.resume else None)
    trainer.save_checkpoint(str(args.output / "checkpoints" / "last.ckpt"))


if __name__ == "__main__":
    main()
