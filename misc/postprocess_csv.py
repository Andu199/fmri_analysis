import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def plot_boxplots(values, connection, disorder, class_names, path):
    plt.boxplot(values, labels=class_names)
    plt.xlabel("Disorder")
    plt.ylabel("Reconstruction Error")
    plt.title(f"Boxplots for connection: {connection} (biomarker for {disorder})")
    plt.savefig(os.path.join(os.path.dirname(path), f"boxplot_{disorder}_{connection}.png"))
    plt.clf()


def clean_df(df, split=None):
    if split is None:
        return df

    indices_path = "/home/ubuntu/Dizertatie/fmri_analysis/misc/val_indices"
    with open(os.path.join(indices_path, "adhd_val.txt"), "r") as f:
        adhd_ids = ["adhd_" + idx.replace("\n", "") for idx in f.readlines()]
    with open(os.path.join(indices_path, "bipolar_val.txt"), "r") as f:
        bipolar_ids = ["bipolar_" + idx.replace("\n", "") for idx in f.readlines()]
    with open(os.path.join(indices_path, "scz_val.txt"), "r") as f:
        schz_ids = ["schz_" + idx.replace("\n", "") for idx in f.readlines()]
    with open(os.path.join(indices_path, "h_val.txt"), "r") as f:
        healthy_ids = ["healthy_" + idx.replace("\n", "") for idx in f.readlines()]

    if split == "val":
        return df[adhd_ids + bipolar_ids + schz_ids + healthy_ids + ["connections"]]

    elif split == "test":
        columns = df.columns
        columns = list(set(columns) - set(adhd_ids + bipolar_ids + schz_ids + healthy_ids))

        return df[columns]

    else:
        raise ValueError("val/test or None!!!")


def postprocess(path, split):
    df = pd.read_csv(path)
    class_names = ["healthy", "schz", "bipolar", "adhd"]  # "hpain", "pain"
    df = clean_df(df, split)
    print(df.columns)

    columns = df.columns
    classes_columns = {
        class_name: [column for column in columns if class_name in column]
        for class_name in class_names
    }

    results_total = {
        f"{class_names[j]}_{class_names[i]}": np.zeros(len(df.index))
        for i in range(len(class_names))
        for j in range(i)
    }
    results_total["connections"] = []

    for idx in df.index:
        connection_name = df["connections"][idx]
        results_total["connections"].append(connection_name)

        classes_values = {
            class_name: df[classes_columns[class_name]].iloc[idx].to_numpy()
            for class_name in class_names
        }
        for i in range(len(class_names)):
            for j in range(i):
                result = mannwhitneyu(classes_values[class_names[j]], classes_values[class_names[i]])[1]
                results_total[f"{class_names[j]}_{class_names[i]}"][idx] = result

    path_output = os.path.join(os.path.dirname(path), f"mannwhitney_{split}_pvalue.csv")
    results_total = pd.DataFrame(results_total)
    results_total.to_csv(path_output, index=False)

    disorders = ["schz", "bipolar", "adhd"]  # "hpain", "pain"
    for disorder in disorders:
        abnormal_connections = results_total[results_total[f'healthy_{disorder}'] < 0.05]['connections'].to_list()
        for connection in abnormal_connections:
            row = df[df['connections'] == connection]

            values = []
            for class_name, columns in classes_columns.items():
                values.append(row[columns].to_numpy().squeeze(axis=0))

            plot_boxplots(values, connection, disorder, class_names, path)


if __name__ == "__main__":
    # for conf_name in ['07_09', '11_02']:
    #     for atlas_name in ['thin', 'thick']:
    #         for corr_name in ['correlation', 'dtw', 'pearson', 'kendall', 'spearman']:
    #             postprocess(f"../outputs/normative/test/{conf_name}_{atlas_name}_{corr_name}/evaluation_per_subject_connection.csv", split="val")

    postprocess(f"../outputs/normative_yeo7/test/version_0/evaluation_per_subject_connection.csv", split="test")
