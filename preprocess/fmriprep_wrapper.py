import os
import subprocess
from argparse import ArgumentParser

from utils.general_utils import config_parser


def create_command_args(configs):
    command_args = []

    # Input and output directory
    if "input_path" not in configs:
        raise ValueError("Input path missing!")
    else:
        input_path = cfg["input_path"]
    if "output_path" not in configs:
        output_path = os.path.join(input_path, "derivatives")
    else:
        output_path = cfg["output_path"]

    command_args.extend([f"{input_path}", f"{output_path}"])

    # Output space
    if "output-spaces" in configs:
        command_args.append("--output-spaces=" + configs["output-spaces"])

    # # Other flags
    if "flags" in configs:
        for flag in configs["flags"]:
            command_args.append(f"--{flag}")

    # License file
    if "fs-license-file" not in configs:
        raise ValueError("Missing fs license file!")
    else:
        command_args.append("--fs-license-file=" + configs["fs-license-file"])

    # Participants
    if "participant_label" in configs:
        arg = "--participant_label="
        for participant in configs["participant_label"]:
            arg += f"{participant} "
        command_args.append(arg[:-1])

    # Tasks
    if "task-id" in configs:
        arg = "--task-id="
        for task in configs["task-id"]:
            arg += f"{task} "
        command_args.append(arg[:-1])

    return command_args


def fmriprep_cli(configs):
    command_args = create_command_args(configs)
    subprocess.run(["fmriprep-docker", *command_args])


if __name__ == '__main__':
    cfg = config_parser(ArgumentParser("Script used to preprocess with fmriprep-docker an input BIDS database"))
    fmriprep_cli(cfg)
