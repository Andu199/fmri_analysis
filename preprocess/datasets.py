import os
from argparse import ArgumentParser

import matplotlib
import nilearn
import pandas as pd
import seaborn as sns
import torch.utils.data
from bids import BIDSLayout
from matplotlib import pyplot as plt
from nilearn import datasets, plotting, input_data
from nilearn.connectome import ConnectivityMeasure
from nilearn.image import image, load_img

from preprocess.constants import CONFOUNDS_DICT
from utils.general_utils import config_parser


class UCLA_LA5c_Dataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sub_names_list = []
        self.sub_data = {}
        self.atlas = None
        self.all_columns_confounds = []

        self.preprocess_data()


    def preprocess_subject_info(self, layout, sub_id):
        func_files = layout.get(subject=sub_id,
                                datatype='func',
                                task=self.config["task"],
                                space='MNI152NLin2009cAsym',
                                suffix='preproc',
                                extension='nii.gz',
                                return_type='file')

        confound_files = layout.get(subject=sub_id,
                                    datatype='func',
                                    task=self.config["task"],
                                    suffix='confounds',
                                    extension="tsv",
                                    return_type='file')

        if len(func_files) != 1 or len(confound_files) != 1:
            return None

        confounds = pd.read_csv(confound_files[0], sep='\t', usecols=self.all_columns_confounds)
        confounds = confounds.values
        confounds = confounds[self.config["clean_arguments"]["tr_drop"]:, :]

        func_img = load_img(func_files[0])
        func_img = func_img[:, :, :, self.config["clean_arguments"]["tr_drop"]:]

        masker = input_data.NiftiLabelsMasker(labels_img=self.atlas,
                                              standardize=self.config["clean_arguments"]["standardize"],
                                              detrend=self.config["clean_arguments"]["detrend"],
                                              low_pass=self.config["clean_arguments"]["low_pass"],
                                              high_pass=self.config["clean_arguments"]["high_pass"],
                                              t_r=self.config["clean_arguments"]["t_r"]
                                              )
        cleaned_img = masker.fit_transform(func_img, confounds)

    def preprocess_data(self):
        print("Parsing and preprocessing all data..")

        # CONFOUND NAMES
        self.all_columns_confounds = []
        for conf_type in self.config["clean_arguments"]["confounds"]:
            columns = CONFOUNDS_DICT.get(conf_type, None)
            if columns is None:
                raise ValueError("Wrong counfounds name in config!")
            self.all_columns_confounds += columns

        # ATLAS
        if self.config["atlas_type"] == "yeo":
            self.atlas = nilearn.datasets.fetch_atlas_yeo_2011()[self.config["atlas_args"]["type"]]
        elif self.config["atlas_type"] == "schaefer":
            self.atlas = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=self.config["atlas_args"]["n_rois"])
        else:
            raise ValueError("Unexpected atlas type in config!")


        # BIDS LAYOUT
        layout = BIDSLayout(self.config["derivatives_dir"], validate=False, config=['bids', 'derivatives'])

        if isinstance(self.config["subjects"], str) and self.config["subjects"] == "ALL":
            self.sub_names_list = [filename.split("-")[1] for filename in os.listdir(self.config["derivatives_dir"])]
        elif isinstance(self.config["subjects"], list):
            self.sub_names_list = self.config["subjects"]
        else:
            raise ValueError("Incorrect type of subject list in config")

        # ACTUAL IMAGE DATA
        for sub_id in self.sub_names_list:
            info = self.preprocess_subject_info(layout, sub_id)
            if info is None:
                # Remove subject from the list
                self.sub_names_list.remove(sub_id)
                continue

            self.sub_data[sub_id] = info

        print("Preprocessing done!")

    def __len__(self):
        return len(self.sub_names_list)

    def __getitem__(self, item):
        # TODO: return also some information regarding the class (either subject id or the actual class)
        return self.sub_data[self.sub_names_list[item]]


if __name__ == '__main__':
    dataset_config = config_parser(ArgumentParser("Run this only for test purposes!"))
    dataset = UCLA_LA5c_Dataset(dataset_config)