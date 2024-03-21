import os
import pickle
from argparse import ArgumentParser
from copy import deepcopy

import nilearn
import numpy as np
import pandas as pd
from bids import BIDSLayout
from nilearn import input_data
from nilearn.image import load_img

from utils.constants import CONFOUNDS_DICT
from utils.general_utils import config_parser


class Container:
    def __init__(self):
        self.sub_names_list = []
        self.sub_data = {}


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.container = Container()
        self.atlas = None
        self.all_columns_confounds = []

    def preprocess(self):
        self.preprocess_data()
        with open(self.config["output_path"], "wb") as f:
            pickle.dump(self.container, f)

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
        func_img = func_img.slicer[:, :, :, self.config["clean_arguments"]["tr_drop"]:]

        masker = input_data.NiftiLabelsMasker(labels_img=deepcopy(self.atlas),
                                              standardize=self.config["clean_arguments"]["standardize"],
                                              detrend=self.config["clean_arguments"]["detrend"],
                                              low_pass=self.config["clean_arguments"]["low_pass"],
                                              high_pass=self.config["clean_arguments"]["high_pass"],
                                              t_r=self.config["clean_arguments"]["t_r"]
                                              )
        cleaned_partial_img = masker.fit_transform(func_img, confounds)

        # fill for the missing labels
        cleaned_img = np.zeros((cleaned_partial_img.shape[0], len(np.unique(self.atlas.get_fdata().astype(int)))))
        cleaned_img[:, np.array(masker.labels_).astype(int)] = cleaned_partial_img

        return cleaned_img

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
            self.atlas = load_img(nilearn.datasets.fetch_atlas_yeo_2011()[self.config["atlas_args"]["type"]])
        elif self.config["atlas_type"] == "schaefer":
            self.atlas = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=self.config["atlas_args"]["n_rois"])
        else:
            raise ValueError("Unexpected atlas type in config!")

        # BIDS LAYOUT
        layout = BIDSLayout(self.config["derivatives_dir"], validate=False, config=['bids', 'derivatives'])

        if isinstance(self.config["subjects"], str) and self.config["subjects"] == "ALL":
            self.container.sub_names_list = [filename.split("-")[1]
                                             for filename in os.listdir(self.config["derivatives_dir"])]
        elif isinstance(self.config["subjects"], list):
            self.container.sub_names_list = self.config["subjects"]
        else:
            raise ValueError("Incorrect type of subject list in config")

        # ACTUAL IMAGE DATA
        list_subs_to_remove = []
        for sub_id in self.container.sub_names_list:
            info = self.preprocess_subject_info(layout, sub_id)
            if info is None:
                # Remove subject from the list
                list_subs_to_remove.append(sub_id)
                continue

            self.container.sub_data[sub_id] = info

        for sub_id in list_subs_to_remove:
            self.container.sub_names_list.remove(sub_id)

        print("Preprocessing done!")


if __name__ == '__main__':
    preprocess_config = config_parser(ArgumentParser("Script used to preprocess a dataset!"))
    preprocessor = Preprocessor(preprocess_config)
    preprocessor.preprocess()
