import pickle
from argparse import ArgumentParser

import torch
from nilearn.connectome import ConnectivityMeasure
from torch.utils.data import DataLoader

from preprocess.preprocess import Container
from utils.general_utils import config_parser


class UCLA_LA5c_Dataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.container = Container()
        self.config = config
        with open(self.config["input_path"], "rb") as f:
            self.container = pickle.load(f)

        if self.config["connectivity_measure_type"] == 'none':
            self.connectivity_measure = None
        else:
            self.connectivity_measure = ConnectivityMeasure(kind=self.config["connectivity_measure_type"])

    def __len__(self):
        return len(self.container.sub_names_list)

    def __getitem__(self, item):
        X = self.container.sub_data[self.container.sub_names_list[item]]
        X = self.connectivity_measure.fit_transform([X])[0]
        return X, self.container.sub_names_list[item]


if __name__ == '__main__':
    dataset_config = config_parser(ArgumentParser("Run this only for test purposes!"))
    dataset = UCLA_LA5c_Dataset(dataset_config)
    dataloader = DataLoader(dataset, batch_size=2)
    for batch_idx, batch in enumerate(dataloader):
        print(batch_idx)
        print(batch)
