import os

import dtw
import numpy as np
import yaml


class ConnectivityDTW:
    def __init__(self):
        pass

    def fit_transform(self, x_list):
        x = x_list[0]
        x = x.T

        connectivity_matrix = np.zeros((x.shape[0], x.shape[0]))
        for i in range(x.shape[0]):
            for j in range(i + 1):
                dist = dtw.dtw(x[i], x[j], distance_only=True).distance
                similarity = 1 / (1 + dist)
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
