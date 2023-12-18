import os

import yaml


def config_parser(parser):
    parser.add_argument("--config_file_path", required=True)
    args = parser.parse_args()

    with open(os.path.join("..\\configs", args.config_file_path), "r") as f:
        config_file = yaml.safe_load(f)

    return config_file
