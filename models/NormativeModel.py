from argparse import ArgumentParser

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.datasets import UCLA_LA5c_Dataset
from models.layers import get_linear_layer
from preprocess.preprocess import Container
from utils.general_utils import config_parser


class NormativeModel(nn.Module):
    def __init__(self, config, input_areas):
        super().__init__()
        self.config = config
        input_dim = (input_areas ** 2 - input_areas) // 2

        fc_list = [get_linear_layer(config, input_dim, config["fc_hidden_size"][0])]
        for idx in range(len(self.config["fc_hidden_size"]) - 1):
            fc_list.append(get_linear_layer(config, config["fc_hidden_size"][idx], config["fc_hidden_size"][idx + 1]))

        fc_list.append(nn.Linear(config["fc_hidden_size"][-1], input_dim))
        self.model = nn.Sequential(*fc_list)

        self.mask = torch.Tensor([[i < j for i in range(input_areas)] for j in range(input_areas)]).to(torch.bool)

    def forward(self, x):
        x = x[:, self.mask]
        return self.model(x)


if __name__ == '__main__':
    model_config = config_parser(ArgumentParser("Run this only for test purposes!"))
    model = NormativeModel(model_config, 18)

    dataset_config = {
        "input_path": "../outputs/dataset_yeo17thick_2024_03_21.pkl",
        "connectivity_measure_type": "correlation",
        "edge_threshold": 0.9,
    }
    dataset = UCLA_LA5c_Dataset(dataset_config)
    dataloader = DataLoader(dataset, batch_size=2)
    for batch_idx, batch in enumerate(dataloader):
        print(batch[1])
        y = model(batch[0])
        print(y)
