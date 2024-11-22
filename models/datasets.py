import pickle
from argparse import ArgumentParser

import numpy as np
import torch
from nilearn.connectome import ConnectivityMeasure
from torch.utils.data import DataLoader

from preprocess.preprocess import Container
from utils.constants import Diagnostic, STRING_TO_DIAGNOSTIC
from utils.general_utils import config_parser, ConnectivityDTW, ConnectivityPandas


class UCLA_LA5c_Dataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.container = Container()
        self.config = config
        with open(self.config["input_path"], "rb") as f:
            self.container = pickle.load(f)

        self.class_label = self.config["class_name"] if "class_name" in self.config.keys() else None
        if self.config["connectivity_measure_type"] == 'none':
            self.connectivity_measure = None
        elif self.config["connectivity_measure_type"] == 'correlation':
            self.connectivity_measure = ConnectivityMeasure(kind=self.config["connectivity_measure_type"])
        elif self.config["connectivity_measure_type"] == 'dtw':
            self.connectivity_measure = ConnectivityDTW()
        elif self.config["connectivity_measure_type"] == 'kendall':
            self.connectivity_measure = ConnectivityPandas(corr_type="kendall")
        elif self.config["connectivity_measure_type"] == 'spearman':
            self.connectivity_measure = ConnectivityPandas(corr_type="spearman")
        elif self.config["connectivity_measure_type"] == 'pearson':
            self.connectivity_measure = ConnectivityPandas(corr_type="pearson")

    def __len__(self):
        return len(self.container.sub_names_list)

    def __getitem__(self, item):
        X = self.container.sub_data[self.container.sub_names_list[item]]
        X = self.connectivity_measure.fit_transform([X.astype(np.float32)])[0]

        if self.class_label is None:
            if self.container.sub_names_list[item][0] == '1':
                y = Diagnostic.HEALTHY.value
            elif self.container.sub_names_list[item][0] == '5':
                y = Diagnostic.SCHZ.value
            elif self.container.sub_names_list[item][0] == '6':
                y = Diagnostic.BIPOLAR.value
            elif self.container.sub_names_list[item][0] == '7':
                y = Diagnostic.ADHD.value
            else:
                raise ValueError("Wrong subject id!")
        else:
            y = STRING_TO_DIAGNOSTIC[self.class_label].value

        return X, y


if __name__ == '__main__':
    dataset_config = config_parser(ArgumentParser("Run this only for test purposes!"))
    dataset = UCLA_LA5c_Dataset(dataset_config)
    dataloader = DataLoader(dataset, batch_size=2)
    for batch_idx, batch in enumerate(dataloader):
        print(batch_idx)
        print(batch)
