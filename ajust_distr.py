import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from neuroCombat import neuroCombat
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.datasets import UCLA_LA5c_Dataset
from test import init_test
from utils.general_utils import config_parser
from preprocess.preprocess import Container


def adjust_distributions(x_data, y_data, y_pain_data):
    # Combine x_data and y_data along the sample (row) dimension.
    # The combined data has shape (320, 136): 320 samples, 136 features.
    combined_data = np.concatenate([x_data, y_data], axis=0)

    # Create batch labels: 'ref' for x_data samples and 'new' for y_data samples
    batch_labels = ['ref'] * x_data.shape[0] + ['new'] * y_data.shape[0]
    covars = pd.DataFrame({'batch': batch_labels})

    # Transpose combined_data to shape (n_features, n_samples) as required by neuroCombat.
    combined_data_T = combined_data.T  # shape becomes (136, 320)

    # Run neuroCombat to harmonize the data.
    combat_results = neuroCombat(dat=combined_data_T,
                                 covars=covars,
                                 batch_col='batch')

    aligned_data_T = combat_results['data']
    aligned_data = aligned_data_T.T  # devine (320, 136)

    # Store y_data after alignement
    y_data_aligned = aligned_data[x_data.shape[0]:, :]

    # Print available keys from neuroCombat output and the estimates sub-dictionary.
    print("neuroCombat output keys:", combat_results.keys())
    estimates = combat_results['estimates']
    print("neuroCombat estimates keys:", estimates.keys())

    # Retrieve gamma and delta from the estimates dictionary.
    # Based on your output, they are stored under 'gamma.star' and 'delta.star'
    if 'gamma.star' in estimates:
        gamma = estimates['gamma.star']
    elif 'gamma_star' in estimates:
        gamma = estimates['gamma_star']
    else:
        raise KeyError("Could not find gamma in estimates: available keys: " + str(estimates.keys()))

    if 'delta.star' in estimates:
        delta = estimates['delta.star']
    elif 'delta_star' in estimates:
        delta = estimates['delta_star']
    else:
        raise KeyError("Could not find delta in estimates: available keys: " + str(estimates.keys()))

    gamma_new = gamma[0, :]  # shape becomes (136,)
    delta_new = delta[0, :]  # shape becomes (136,)

    # Reshape gamma_new and delta_new to (136, 1) for proper broadcasting.
    gamma_new = gamma_new[:, np.newaxis]  # now shape: (136, 1)
    delta_new = delta_new[:, np.newaxis]  # now shape: (136, 1)

    # Compute the overall means (alpha) per feature from the combined (transposed) data.
    alpha = combined_data_T.mean(axis=1)  # shape: (136,)

    # Process y_new_data: transpose it to have shape (n_features, n_samples) as required by the transformation.
    y_pain_data_T = y_pain_data.T  # shape becomes (136, 20)

    # Center the new data using alpha.
    centered = y_pain_data_T - alpha[:, np.newaxis]  # shape: (136, 20)

    # Apply the transformation to obtain the adjusted (harmonized) new data.
    pain_adjusted_data_T = (centered - gamma_new) / delta_new  # shape: (136, 20)

    # Transpose back to original orientation: (n_samples, n_features)
    pain_adjusted_data = pain_adjusted_data_T.T  # now shape: (20, 136)
    # pain_adjusted_data coresponds to pain class

    # Print shapes of all datasets to ensure consistency.

    x_means = x_data.mean(axis=1)  # shape: (300,)
    y_means = y_data.mean(axis=1)  # shape: (20,)
    y_data_aligned_means = y_data_aligned.mean(axis=1)
    y_pain_means = y_pain_data.mean(axis=1)  # shape: (20,)
    adjusted_pain_means = pain_adjusted_data.mean(axis=1)  # shape: (20,)

    # Create a figure with two subplots.
    plt.figure(figsize=(14, 6))

    # Left subplot: Histogram of sample means.
    plt.subplot(1, 2, 1)
    plt.hist(x_means, bins=30, alpha=0.6, label='x_data (ref)')
    plt.hist(y_means, bins=30, alpha=0.6, label='y_data (old)')
    plt.hist(y_data_aligned_means, bins=30, alpha=0.6, label='y_data (aligned)')
    plt.hist(y_pain_means, bins=30, alpha=0.6, label='y_pain_data (pre-align)')
    plt.hist(adjusted_pain_means, bins=30, alpha=0.6, label='y_pain_data (post-align)')
    plt.xlabel('Mean across features')
    plt.ylabel('Frequency')
    plt.title('Histogram of Sample Means')
    plt.legend()

    # Right subplot: Boxplot for a side-by-side comparison.
    plt.subplot(1, 2, 2)
    data_to_plot = [x_means, y_means, y_data_aligned_means, y_pain_means, adjusted_pain_means]
    plt.boxplot(data_to_plot,
                labels=['x_data (ref)', 'y (pre-align)', 'y (aligned)', 'pain (pre-align)', 'pain (aligned)'])
    plt.title('Boxplot of Sample Means')

    plt.tight_layout()
    plt.savefig("plot_adjusted_data.png")
    plt.clf()

    return y_data_aligned, pain_adjusted_data

if __name__ == "__main__":
    # np.random.seed(42)
    #
    # # Generate x_data: 300 samples, 136 features (mimics UCLA distribution - healty)
    # x_data = np.random.normal(loc=0, scale=1, size=(300, 136))
    #
    # # Generate y_data: 20 samples, 136 features (mimics pain distribution - healty)
    # y_data = np.random.normal(loc=0.2, scale=1, size=(20, 136))
    #
    # # Generate y_new_data: 20 samples, 136 features (mimics pain distribution - pain)
    # y_pain_data = np.random.normal(loc=0.2, scale=1, size=(20, 136))
    # # mofify some features to look like pain data
    # shift_feature_indices = [10, 11, 12, 13]
    # y_pain_data[:, shift_feature_indices] += 0.3
    #
    # #  Align y_data to x_data
    # y_data_aligned, pain_adjusted_data = adjust_distributions(x_data, y_data, y_pain_data)

    config = config_parser(ArgumentParser("Side testing script. Used for data distribution adjustment"))
    model, dataloaders, overall_metric, per_connection_metrics = init_test(config)

    path = (config["ds_paths"]['healthy']).replace("test.pkl", "train.pkl")
    config_dataset = {
        "input_path": path,
        "connectivity_measure_type": config["connectivity_measure_type"],
        "edge_threshold": config["edge_threshold"],
        "class_name": "healthy"
    }
    dataset = UCLA_LA5c_Dataset(config_dataset)
    dl = DataLoader(dataset, config["batch_size"], shuffle=False)
    dataloaders["healthy_train"] = dl

    overall_metric_results = {}
    per_connection_metric_results = {}

    os.makedirs(os.path.join(config["output_dir"], "plots"), exist_ok=True)

    x_data_train = []
    x_data_test = []
    y_data = []
    y_pain_data = []
    mask = model.mask
    for (x, _) in dataloaders["healthy_train"]:
        x = x[:, mask]
        x_data_train.append(x.squeeze(0).cpu().numpy())
    for (x, _) in dataloaders["healthy"]:
        x = x[:, mask]
        x_data_test.append(x.squeeze(0).cpu().numpy())
    for (x, _) in dataloaders["hpain"]:
        x = x[:, mask]
        y_data.append(x.squeeze(0).cpu().numpy())
    for (x, _) in dataloaders["pain"]:
        x = x[:, mask]
        y_pain_data.append(x.squeeze(0).cpu().numpy())

    x_data_train = np.stack(x_data_train, axis=0)
    x_data_test = np.stack(x_data_test, axis=0)
    x_data = np.concatenate((x_data_train, x_data_test), axis=0)
    print(x_data.shape)
    y_data = np.stack(y_data, axis=0)
    y_pain_data = np.stack(y_pain_data, axis=0)

    y_data_aligned, pain_adjusted_data = adjust_distributions(x_data, y_data, y_pain_data)

    datasets_to_infer = {
        "healthy": torch.from_numpy(x_data_test).to(torch.float32),
        "hpain": torch.from_numpy(y_data_aligned).to(torch.float32),
        "pain": torch.from_numpy(pain_adjusted_data).to(torch.float32)
    }

    for name in ['adhd', 'bipolar', 'schz']:
        data_name = []
        for (x, _) in dataloaders[name]:
            x = x[:, mask]
            data_name.append(x.squeeze(0).cpu().numpy())
        data_name = torch.from_numpy(np.stack(data_name, axis=0)).to(torch.float32)
        datasets_to_infer[name] = data_name

    for name, dl in datasets_to_infer.items():
        overall_metric.reset()

        with torch.no_grad():
            for id_sub, (x) in enumerate(dl):
                x = x.unsqueeze(0)
                for metric in per_connection_metrics:
                    metric.reset()

                pred = model.model(x)
                y = x

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
