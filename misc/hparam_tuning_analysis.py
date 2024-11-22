import json
import os

import pandas as pd

if __name__ == "__main__":
    confounds_name = ["07_09", "11_02"]
    atlas_name = ["thin", "thick"]
    connectivity_name = ["correlation", "dtw", "kendall", "spearman", "pearson"]

    path = "/home/ubuntu/Dizertatie/fmri_analysis/outputs/normative/test"
    filename = "mannwhitney_test_pvalue.csv"

    p_thr = 0.05

    relevant_connections = {}
    freq_connections = {}
    scores_experiments = {}

    for conf_name in confounds_name:
        for at_name in atlas_name:
            for conn_name in connectivity_name:
                experiment_name = f"{conf_name}_{at_name}_{conn_name}"
                filepath = os.path.join(path, experiment_name, filename)
                rel_conn = []
                scores_experiments[experiment_name] = 0

                df = pd.read_csv(filepath)
                wanted_keys = [colname for colname in df.columns if "healthy" in colname]
                wanted_keys.append("connections")
                df = df[wanted_keys]

                for colname in df.columns:
                    if colname == "connections":
                        continue
                    connection_names = (df[df[colname] <= p_thr])['connections'].to_list()
                    rel_conn.extend(connection_names)

                relevant_connections[experiment_name] = list(set(rel_conn))
                for rel_conn_name in list(set(rel_conn)):
                    if rel_conn_name not in freq_connections:
                        freq_connections[rel_conn_name] = 0
                    freq_connections[rel_conn_name] += 1

    freq_connections = dict(sorted(freq_connections.items(), key=lambda item: item[1], reverse=True))

    for idx, key in enumerate(freq_connections):
        if freq_connections[key] < 10:  # vary this.
            continue
        score = freq_connections[key]
        for exp in relevant_connections.keys():
            if key in relevant_connections[exp]:
                scores_experiments[exp] += score

    scores_experiments = dict(sorted(scores_experiments.items(), key=lambda item: item[1], reverse=True))
    with open("best_experiments.json", "w") as f:
        json.dump(scores_experiments, f)

    with open("freq_vector.json", "w") as f:
        json.dump(freq_connections, f)
