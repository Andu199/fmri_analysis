import os.path
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


def plot_reconstructions(per_connection_metric_results, config):
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
        plt.savefig(os.path.join(config["output_dir"], "plots",f"heatmap_{name}_wo_subtraction.png"))
        plt.clf()

    mean_healthy = np.mean([all_matrices[key] for key in all_matrices.keys() if "healthy" in key], axis=0)

    # Plot with subtraction
    for name, matrix in all_matrices.items():
        matrix -= mean_healthy
        sns.heatmap(matrix)
        plt.savefig(os.path.join(config["output_dir"], "plots", f"heatmap_{name}_with_subtraction.png"))
        plt.clf()


def test(config):
    model, dataloaders, overall_metric, per_connection_metrics = init_test(config)
    overall_metric_results = {}
    per_connection_metric_results = {}

    os.makedirs(os.path.join(config["output_dir"], "plots"), exist_ok=True)

    for name, dl in dataloaders.items():
        overall_metric.reset()

        with torch.no_grad():
            for id_sub, (x, _) in enumerate(dl):
                for metric in per_connection_metrics:
                    metric.reset()

                pred = model(x)
                y = x[:, model.mask]

                overall_metric.update(pred, y)
                for idx, metric in enumerate(per_connection_metrics):
                    metric.update(pred[:, idx], y[:, idx])

                per_connection_metric_results[f"{name}_{id_sub}"] = []
                for metric in per_connection_metrics:
                    mse_result = metric.compute()
                    per_connection_metric_results[f"{name}_{id_sub}"].append(mse_result.item())

        overall_mse_result = overall_metric.compute()
        overall_metric_results[name] = overall_mse_result.item()

    pd.DataFrame(
        np.array(list(overall_metric_results.values())).reshape(1, -1), columns=list(overall_metric_results.keys())
    ).to_csv(os.path.join(config["output_dir"], "evaluation_overall.csv"), index=False)

    df = pd.DataFrame(
        np.array(list(per_connection_metric_results.values())).T, columns=list(per_connection_metric_results.keys())
    )
    networks = ['vis_1', 'vis_2', 'mot_1', 'mot_2', 'dan_2', 'dan_1', 'van_1', 'fp_1', 'lim_1', 'lim_2', 'fp_2', 'fp_3',
                'fp_4', 'mot_3', 'dmn_3', 'dmn_1', 'dmn_2']
    df["connections"] = [f"{networks[i]}_{networks[j]}"
                         for i in range(len(networks)) for j in range(i)]

    df.to_csv(os.path.join(config["output_dir"], "evaluation_per_subject_connection.csv"), index=False)
    plot_reconstructions(per_connection_metric_results, config)


if __name__ == "__main__":
    test_config = config_parser(ArgumentParser("Main testing script. It gets a YAML file with configurations for:"
                                               "model checkpoint, dataset paths, etc."))
    test(test_config)
