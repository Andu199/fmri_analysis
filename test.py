from argparse import ArgumentParser

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchmetrics
from matplotlib import pyplot as plt
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
        overall_metric = torchmetrics.MeanSquaredError()
        input_dim = (config["input_areas"] ** 2 - config["input_areas"]) // 2
        per_connection_metrics = [
            torchmetrics.MeanSquaredError() for _ in range(input_dim)
        ]
    elif config["metric"] == "mae":
        overall_metric = torchmetrics.MeanAbsoluteError()
        input_dim = (config["input_areas"] ** 2 - config["input_areas"]) // 2
        per_connection_metrics = [
            torchmetrics.MeanAbsoluteError() for _ in range(input_dim)
        ]
    else:
        raise ValueError(config["metric"] + " not yet implemented!")

    return model, dataloaders, overall_metric, per_connection_metrics


def plot_reconstructions(per_connection_metric_results):
    all_matrices = {}
    for name, values in per_connection_metric_results.items():
        dim = len(values)
        areas = (1 + int(np.sqrt(1 + 8 * dim))) // 2  # y = (x^2 - x) / 2 => x^2 - x - 2*y = 0
        matrix = np.zeros((areas, areas))

        index = 0
        for i in range(areas):
            for j in range(i):
                matrix[i, j] = values[index]
                matrix[j, i] = values[index]
                index += 1
            matrix[i, i] = 1.0

        all_matrices[name] = matrix

    # Plot without subtraction
    for name, matrix in all_matrices.items():
        sns.heatmap(matrix)
        plt.savefig(f"outputs/heatmap_{name}_wo_subtraction.png")
        plt.clf()

    # Plot wiht subtraction
    for name, matrix in all_matrices.items():
        if name == "healthy":
            continue
        matrix -= all_matrices["healthy"]
        sns.heatmap(matrix)
        plt.savefig(f"outputs/heatmap_{name}_with_subtraction.png")
        plt.clf()


def test(config):
    model, dataloaders, overall_metric, per_connection_metrics = init_test(config)
    overall_metric_results = {}
    per_connection_metric_results = {name: [] for name in dataloaders.keys()}

    for name, dl in dataloaders.items():
        overall_metric.reset()
        for metric in per_connection_metrics:
            metric.reset()

        with torch.no_grad():
            for x, _ in dl:
                pred = model(x)
                y = x[:, model.mask]

                overall_metric.update(pred, y)
                for idx, metric in enumerate(per_connection_metrics):
                    metric.update(pred[:, idx], y[:, idx])

        overall_mse_result = overall_metric.compute()
        overall_metric_results[name] = overall_mse_result.item()

        for metric in per_connection_metrics:
            mse_result = metric.compute()
            per_connection_metric_results[name].append(mse_result.item())

    pd.DataFrame(
        np.array(list(overall_metric_results.values())).reshape(1, -1), columns=list(overall_metric_results.keys())
    ).to_csv("outputs/normative/test/version_1/evaluation_overall.csv", index=False)
    pd.DataFrame(
        np.array(list(per_connection_metric_results.values())).T, columns=list(per_connection_metric_results.keys())
    ).to_csv("outputs/normative/test/version_1/evaluation_per_connection.csv", index=False)
    plot_reconstructions(per_connection_metric_results)


if __name__ == "__main__":
    test_config = config_parser(ArgumentParser("Main testing script. It gets a YAML file with configurations for:"
                                               "model checkpoint, dataset paths, etc."))
    test(test_config)
