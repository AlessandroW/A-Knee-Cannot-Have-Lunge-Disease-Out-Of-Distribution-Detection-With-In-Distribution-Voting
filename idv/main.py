from argparse import ArgumentParser
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from idv.datamodule import DataModule
from idv.model import Model


def add_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group("Main")
    parser.add_argument(
        "--tensorboard_dir", default="run", help="The subdir in TensorBoad."
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--early_stopping_patience", default=3, type=int)
    return parent_parser


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = add_argparse_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser = Model.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    dict_args = vars(args)

    pl.seed_everything(dict_args["seed"], workers=True)
    datamodule = DataModule.from_argparse_args(args)
    model = Model.from_argparse_args(args)
    logger = TensorBoardLogger("experiments", name=dict_args["tensorboard_dir"],
                               default_hp_metric=dict_args["default_hp_metric"])

    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        save_top_k=50, monitor="loss/val_epoch", verbose=True, save_last=True, mode="min"
    )
    early_stop_callback = EarlyStopping(monitor="loss/val_epoch", min_delta=0.00, patience=dict_args["early_stopping_patience"], verbose=True, mode="min")
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model, datamodule)
