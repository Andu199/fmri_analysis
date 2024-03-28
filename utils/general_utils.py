import os

import yaml


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
