import hydra
import os
from argparse import Namespace
from omegaconf import OmegaConf
from pathlib import Path
import warnings

import torch
import pytorch_lightning as pl
import yaml
import numpy as np

from lightning_modules import LigandPocketDDPM

import utils


def merge_args_and_yaml(args, config_dict):
    arg_dict = args if isinstance(args, dict) else args.__dict__
    for key, value in config_dict.items():
        if key in arg_dict:
            warnings.warn(f"Command line argument '{key}' (value: "
                          f"{arg_dict[key]}) will be overwritten with value "
                          f"{value} provided in the config file.")
        if isinstance(value, dict):
            arg_dict[key] = Namespace(**value)
        else:
            arg_dict[key] = value

    return args


def merge_configs(config, resume_config):
    for key, value in resume_config.items():
        if isinstance(value, Namespace):
            value = value.__dict__
        if key in config and config[key] != value:
            warnings.warn(f"Config parameter '{key}' (value: "
                          f"{config[key]}) will be overwritten with value "
                          f"{value} from the checkpoint.")
        config[key] = value
    return config


# ------------------------------------------------------------------------------
# Training
# ______________________________________________________________________________
@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(args):
    utils.load_and_set_env_variables()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    assert 'resume' not in config

    # Get main config
    if args.resume is not None and os.path.exists(Path(args.resume)):
        ckpt_path = Path(args.resume)
    else:
        ckpt_path = None
    if ckpt_path is not None:
        resume_config = torch.load(
            ckpt_path, map_location=torch.device('cpu'))['hyper_parameters']

        config = merge_configs(config, resume_config)

    args = Namespace(**merge_args_and_yaml(OmegaConf.to_container(args, resolve=True), config))

    out_dir = Path(args.logdir, args.run_name)
    histogram_file = Path(args.datadir, 'size_distribution.npy')
    histogram = np.load(histogram_file).tolist()
    pl_module = LigandPocketDDPM(
        outdir=out_dir,
        dataset=args.dataset,
        datadir=args.datadir,
        batch_size=args.batch_size,
        lr=args.lr,
        net_dynamics_params=args.net_dynamics_params,
        diffusion_params=args.diffusion_params,
        num_workers=args.num_workers,
        augment_noise=args.augment_noise,
        augment_rotation=args.augment_rotation,
        clip_grad=args.clip_grad,
        eval_epochs=args.eval_epochs,
        eval_params=args.eval_params,
        visualize_sample_epoch=args.visualize_sample_epoch,
        visualize_chain_epoch=args.visualize_chain_epoch,
        auxiliary_loss=args.auxiliary_loss,
        loss_params=args.loss_params,
        mode=args.mode,
        node_histogram=histogram,
        pocket_representation=args.pocket_representation,
        gcpnet_config=getattr(args, "gcpnet", None),
    )

    logger = pl.loggers.WandbLogger(
        save_dir=args.logdir,
        project='ligand-pocket-ddpm',
        group=args.dataset,
        name=args.run_name,
        id=args.run_name,
        resume='must' if ckpt_path is not None else False,
        entity=args.wandb_params.entity,
        mode=args.wandb_params.mode,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(out_dir, 'checkpoints'),
        filename="best-model-epoch={epoch:02d}",
        monitor="loss/val",
        # auto_insert_metric_name=False,
        save_top_k=1,
        save_last=True,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=args.enable_progress_bar,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        accelerator='gpu', devices=args.gpus,
        strategy=args.strategy if hasattr(args, "strategy") else ('ddp' if args.gpus > 0 else None),
    )

    trainer.fit(model=pl_module, ckpt_path=ckpt_path)

    # # run test set
    # result = trainer.test(ckpt_path='best')


if __name__ == "__main__":
    main()
