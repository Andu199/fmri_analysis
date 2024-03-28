import os
from argparse import ArgumentParser
from copy import deepcopy

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader

from models.NormativeModel import NormativeModel
from models.datasets import UCLA_LA5c_Dataset
from utils.general_utils import config_parser


def init_train(config):
    if config["model_name"] == "normative":
        model = NormativeModel(config)
    else:
        raise ValueError(config["model_name"] + " not yet implemented!")

    if config["dataset_name"] == "ucla":
        train_config = deepcopy(config)
        train_config["input_path"] = train_config["train_path"]
        train_dataset = UCLA_LA5c_Dataset(train_config)

        val_config = deepcopy(config)
        val_config["input_path"] = val_config["val_path"]
        val_dataset = UCLA_LA5c_Dataset(val_config)
    else:
        raise ValueError(config["dataset_name"] + " not yet implemented!")

    train_dl = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    logger_path = os.path.join("../outputs", config["model_name"], "logs")
    if config["logger"] == "tensorboard":
        logger = TensorBoardLogger(logger_path, name="my_experiment")
    elif config["logger"] == "csv":
        logger = CSVLogger(logger_path, name="my_experiment")
    else:
        raise ValueError(config["logger"] + " not yer implemented!")

    ckpts_path = os.path.join("../outputs", config["model_name"], "ckpts")
    checkpoint_callback = ModelCheckpoint(
        monitor=config["ckpt_args"]["metric"],
        dirpath=ckpts_path,
        filename='model-{epoch:02d}',
        mode=config["ckpt_args"]["mode"],
    )

    return model, train_dl, val_dl, logger, checkpoint_callback


def train(config):
    model, train_dl, val_dl, logger, checkpoint_callback = init_train(config)
    trainer = L.Trainer(max_epochs=config["max_epochs"], logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    train_config = config_parser(ArgumentParser("Main training script. It gets a YAML file with configurations for:"
                                                "model, data, training process and other auxiliary tools"))
    train(train_config)
