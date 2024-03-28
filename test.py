from argparse import ArgumentParser

import pandas as pd
import torch
import torchmetrics
from torch.utils.data import DataLoader

from models.NormativeModel import NormativeModel
from models.datasets import UCLA_LA5c_Dataset
from preprocess.preprocess import Container
from utils.general_utils import config_parser


def init_test(config):
    if config["model_name"] == "normative":
        model = NormativeModel(config)
    else:
        raise ValueError(config["model_name"] + " not yet implemented!")
    state_dict = torch.load(config["model_path"])["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    dataloaders = {}
    for name, path in config["ds_paths"].items():
        if config["dataset_name"] == "ucla":
            config_dataset = {
                "input_path": path,
                "connectivity_measure_type": config["connectivity_measure_type"],
                "edge_threshold": config["edge_threshold"],
            }
            dataset = UCLA_LA5c_Dataset(config_dataset)
            dl = DataLoader(dataset, config["batch_size"], shuffle=False)
            dataloaders[name] = dl

    if config["metric"] == "mse":
        metric = torchmetrics.MeanSquaredError()
    else:
        raise ValueError(config["metric"] + " not yet implemented!")

    return model, dataloaders, metric


def test(config):
    model, dataloaders, metric = init_test(config)
    metric_results = {}

    for name, dl in dataloaders.items():
        metric.reset()
        with torch.no_grad():
            for x, _ in dl:
                pred = model(x)
                y = x[:, model.mask]
                metric.update(pred, y)
        mse_result = metric.compute()
        metric_results[name] = mse_result.item()

    pd.DataFrame(metric_results).to_csv("outputs/evaluation.csv")


if __name__ == "__main__":
    test_config = config_parser(ArgumentParser("Main testing script. It gets a YAML file with configurations for:"
                                               "model checkpoint, dataset paths, etc."))
    test(test_config)
