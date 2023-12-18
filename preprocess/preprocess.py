from argparse import ArgumentParser

import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
from nilearn import datasets, plotting, input_data
from nilearn.connectome import ConnectivityMeasure
from nilearn.image import image

from utils.general_utils import config_parser


def load_clean_image(path, clean=False):
    assert isinstance(path, str)
    data = image.load_img(path)
    assert len(data.shape) == 4

    if clean:
        data = image.smooth_img(data, fwhm=10)
        data = image.clean_img(data, standardize=True, detrend=True)
        plotting.plot_epi(image.mean_img(data))
        plt.savefig("mean_img_cleaned.jpg")
        plt.clf()
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    return data


def save_results_connectome(corr_mat, coords, labels, edge_threshold):
    if labels is None:
        sns.heatmap(corr_mat)
    else:
        sns.heatmap(corr_mat, xticklabels=labels, yticklabels=labels)

    plt.title("Corerlation Matrix")
    plt.savefig("correlation_matrix.jpg", dpi=300)
    plt.clf()

    plotting.plot_connectome(
        corr_mat,
        coords,
        edge_threshold=edge_threshold
    )
    plt.savefig("connectome_plot.jpg")
    plt.clf()

    connectome_plot = plotting.view_connectome(
        corr_mat,
        coords,
        edge_threshold=edge_threshold
    )
    connectome_plot.save_as_html('connectome_viewer.html')


def connectome_calculator(configs):
    # Load data
    data = load_clean_image(configs["func_image_path"])

    # Define a region of interest (ROI)
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=configs["atlas_n_rois"])
    roi_masker = input_data.NiftiLabelsMasker(labels_img=atlas.maps)
    roi_img = roi_masker.fit_transform(data)

    # Compute connectome
    cm = ConnectivityMeasure(kind=configs["connectivity_measure_type"])
    corr_mat = cm.fit_transform([roi_img])
    corr_mat = corr_mat.squeeze()

    coords = plotting.find_parcellation_cut_coords(atlas.maps)
    save_results_connectome(corr_mat, coords, None, configs["edge_threshold"])


if __name__ == '__main__':
    config = config_parser(ArgumentParser("Script with multiple functionalities implemented with nilearn which compute"
                                          "inputs for ML models"))
    connectome_calculator(config)
