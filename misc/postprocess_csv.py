import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


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
    pd.DataFrame(results_total).to_csv(path_output, index=False)


if __name__ == "__main__":
    postprocess("../outputs/normative/test/version_2/evaluation_per_subject_connection.csv")
