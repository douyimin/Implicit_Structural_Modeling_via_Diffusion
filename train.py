"""
Training entry point for the conditional diffusion implicit-structural-modeling
model used in "Implicit Structural Modeling via Generative Diffusion Framework".

All paths and hyper-parameters are passed on the command line; nothing is
hard-coded. The training and validation datasets are produced by the synthesis
pipeline described in the paper (see README and the data release).

Example
-------
    python train.py \
        --config       ./models/cldm_v21.yaml \
        --resume       weights/init.ckpt \
        --train-data   ../fault2rgtdata \
        --val-data     val_data \
        --batch-size   16 \
        --lr           1e-5 \
        --gpus         0,1
"""

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(description="Train the conditional diffusion model.")
    p.add_argument('--config', default='./models/cldm_v21.yaml',
                   help='Model config YAML.')
    p.add_argument('--resume', required=True,
                   help='Checkpoint to initialise / resume from.')
    p.add_argument('--train-data', required=True,
                   help='Directory of the (synthetic) training dataset.')
    p.add_argument('--val-data', default='val_data',
                   help='Directory of the validation dataset.')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--logger-freq', type=int, default=500,
                   help='Image-logging frequency (in steps).')
    p.add_argument('--num-workers', type=int, default=12)
    p.add_argument('--sd-locked', action='store_true',
                   help='Freeze the SD decoder (default: trainable).')
    p.add_argument('--only-mid-control', action='store_true')
    p.add_argument('--gpus', default='0',
                   help='Comma-separated CUDA device ids, e.g. "0,1".')
    return p.parse_args()


def main():
    args = parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from pytorch_lightning.strategies.ddp import DDPStrategy

    import share  # noqa: F401  (disables verbosity; optional sliced attention)
    from cldm.logger import ImageLogger
    from cldm.model import create_model, load_state_dict
    from struc2rgtDataset import seisDataset
    from valDataset import valDataset

    # Load on CPU first; Lightning moves it to the GPU(s) automatically.
    model = create_model(args.config).cpu()
    msg = model.load_state_dict(load_state_dict(args.resume, location='cpu'), strict=False)
    print(msg)

    model.learning_rate = args.lr
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control

    train_loader = DataLoader(
        seisDataset(args.train_data),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        valDataset(args.val_data),
        num_workers=0, batch_size=1, shuffle=False)

    logger = ImageLogger(batch_frequency=args.logger_freq)
    trainer = pl.Trainer(
        precision=16,
        callbacks=[logger],
        strategy=DDPStrategy(static_graph=True, find_unused_parameters=True))

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
