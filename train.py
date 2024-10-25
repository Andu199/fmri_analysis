import json
import os
from argparse import ArgumentParser
from copy import deepcopy
from itertools import product

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader

from models.NormativeModel import NormativeModel
from models.datasets import UCLA_LA5c_Dataset
from preprocess.preprocess import Container
from utils.general_utils import config_parser


def init_train(config):
    # MODEL
    if config["model_name"] == "normative":
        model = NormativeModel(config)
    else:
        raise ValueError(config["model_name"] + " not yet implemented!")

    # DATA
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

    # LOGGER
    logger_path = os.path.join("outputs", config["model_name"], "logs")
    if config["logger"] == "tensorboard":
        logger = TensorBoardLogger(logger_path, name="my_experiment")
    elif config["logger"] == "csv":
        logger = CSVLogger(logger_path, name="my_experiment")
    else:
        raise ValueError(config["logger"] + " not yer implemented!")

    # CALLBACKS
    callbacks = []
    if "ckpt_args" in config:
        callbacks.append(
            ModelCheckpoint(
                monitor=config["ckpt_args"]["metric"],
                dirpath=os.path.join("outputs", config["model_name"], "ckpts"),
                filename='model-{epoch:02d}',
                mode=config["ckpt_args"]["mode"],
            )
        )

    if "early_stopping" in config:
        callbacks.append(
            EarlyStopping(
                monitor=config["early_stopping"]["metric"],
                mode=config["early_stopping"]["mode"],
                min_delta=float(config["early_stopping"]["min_delta"]),
                patience=int(config["early_stopping"]["patience"])
            )
        )

    return model, train_dl, val_dl, logger, callbacks


def train(config):
    model, train_dl, val_dl, logger, callbacks = init_train(config)
    trainer = L.Trainer(max_epochs=config["max_epochs"], logger=logger, callbacks=callbacks)
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


def hparams_tuning(config):
    if "tune" not in config or not isinstance(config["tune"], list) or len(config["tune"]) == 0:
        raise ValueError("Set the \"tune\" parameter and add all the keys that need to be tuned (grid search procedure)")

    params_to_finetune = {}
    for hparam, values in config.items():
        if hparam == "tune":
            continue
        if hparam in config["tune"]:
            if hparam not in config:
                raise ValueError(f"Unknown hyperparameter: {hparam}")
            if not isinstance(config[hparam], list):
                raise ValueError(f"Entry for hyperparameter {hparam} needs to be a list")

            params_to_finetune[hparam] = config[hparam]
        else:
            params_to_finetune[hparam] = [config[hparam]]

    param_combinations = list(product(*params_to_finetune.values()))
    for combination in param_combinations:
        hparams = dict(zip(params_to_finetune.keys(), combination))
        with open(os.path.join("outputs", config["model_name"], "hparams.txt"), "a") as f:
            f.write(json.dumps(hparams))
            f.write("\n\n")
        train(hparams)


if __name__ == "__main__":
    train_config = config_parser(ArgumentParser("Main training script. It gets a YAML file with configurations for:"
                                                "model, data, training process and other auxiliary tools"))

    ### SIMPLE TRAIN
    train(train_config)

    ### GRID SEARCH
    # # change as it is needed
    # train_config["weight_decay"] = [0, 1e-5, 1e-4]
    # train_config["lr"] = [1e-4, 5e-4]
    # train_config["tune"] = ["weight_decay", "lr"]  # add all keys that have been modified for tuning
    # train_config["epochs"] = 1
    #
    # hparams_tuning(train_config)
