import os

import pandas as pd
from joblib import Parallel, delayed

from DTW_for_fMRI.dtw_connectivity_parallel import dtw
import numpy as np
import yaml


def compute_similarity_from_distance(dist, other_params=None, sim_type=0):
    if sim_type == -1:
        return dist
    elif sim_type == 0:
        return 1 / (1 + dist)
    elif sim_type == 1:
        if other_params is None or "lambda" not in other_params or not isinstance(other_params["lambda"], int):
            raise ValueError("Wrong other_params value!")

        return np.exp(-other_params["lambda"] * dist)
    else:
        raise ValueError("Wrong sim_type")


class ConnectivityPandas:
    def __init__(self, corr_type="pearson"):
        self.corr_type = corr_type

    def fit_transform(self, x_list):
        x = x_list[0]
        x = pd.DataFrame(x)
        connectivity_matrix = x.corr(self.corr_type)

        return np.array([connectivity_matrix]).astype(np.float32)


class ConnectivityDTW:
    def __init__(self, sim_type=0, window_size=5, lambda_value=None):
        self.sim_type = sim_type
        self.window_size = window_size
        self.other_params = {
            "lambda": lambda_value,
        }

    def fit_transform(self, x_list):
        x = x_list[0]
        x = x.T

        connectivity_matrix = np.zeros((x.shape[0], x.shape[0]))

        # Parallelize the DTW calculations
        def calculate_similarity(i, j):
            dist = dtw(x[i], x[j], w=self.window_size, scale_with_timeseries_length=True)
            sim = compute_similarity_from_distance(dist, other_params=self.other_params, sim_type=self.sim_type)
            return i, j, sim

        results = Parallel(n_jobs=-1)(
            delayed(calculate_similarity)(i, j) for i in range(x.shape[0]) for j in range(i + 1))

        for i, j, similarity in results:
            connectivity_matrix[i, j] = similarity
            connectivity_matrix[j, i] = similarity

        return np.array([connectivity_matrix]).astype(np.float32)


def config_parser(parser):
    parser.add_argument("--config_file_path", required=True)
    args = parser.parse_args()

    try:
        with open(os.path.join("..\\configs", args.config_file_path), "r") as f:
            config_file = yaml.safe_load(f)
    except FileNotFoundError:
        with open(os.path.join("configs", args.config_file_path), "r") as f:
            config_file = yaml.safe_load(f)

    return config_file


def str2bool(str_flag) -> bool:
    if isinstance(str_flag, bool):
        return str_flag
    if isinstance(str_flag, int):
        return str_flag != 0
    if not isinstance(str_flag, str):
        raise ValueError(f"Got invalid argument type {type(str_flag)}")

    if str_flag.lower() in ['true', "t", '1', 'y', 'yes']:
        return True
    elif str_flag.lower() in ["false", "f", "0", "n", "no"]:
        return False
    raise ValueError(f"Got invalid argument {str_flag}")
