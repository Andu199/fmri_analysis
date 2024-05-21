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


def postprocess(path):
    df = pd.read_csv(path)
    class_names = ["healthy", "schz", "bipolar", "adhd"]
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

    path_output = os.path.join(os.path.dirname(path), "mannwhitney_test_pvalue.csv")
    results_total = pd.DataFrame(results_total)
    results_total.to_csv(path_output, index=False)

    disorders = ["schz", "bipolar", "adhd"]
    for disorder in disorders:
        abnormal_connections = results_total[results_total[f'healthy_{disorder}'] <= 0.05]['connections'].to_list()
        for connection in abnormal_connections:
            row = df[df['connections'] == connection]

            values = []
            for class_name, columns in classes_columns.items():
                values.append(row[columns].to_numpy().squeeze(axis=0))

            plot_boxplots(values, connection, disorder, class_names, path)


if __name__ == "__main__":
    postprocess("../outputs/normative/test/version_2/evaluation_per_subject_connection.csv")
