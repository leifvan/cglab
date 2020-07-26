import argparse

import imageio
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
from matplotlib.ticker import FormatStrFormatter

from gradient_directions import get_n_equidistant_angles_and_intervals
from methods import apply_transform, estimate_transform_by_minimizing_energy, \
    estimate_dense_displacements_from_memberships, estimate_linear_transform
from patches import find_promising_patch_pairs
from utils import plot_diff, GifExporter, pad_slices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('feature_map_path')
    args = parser.parse_args()

    # -------------------------------
    # load feature map and preprocess
    # -------------------------------

    feature_map = imageio.imread(args.feature_map_path).astype(np.float32)

    if len(feature_map.shape) == 3:
        feature_map = np.mean(feature_map, axis=2)

    feature_map = skimage.transform.downscale_local_mean(feature_map, (4, 4))
    feature_map /= feature_map.max()
    feature_map[feature_map > 0.5] = 1
    feature_map[feature_map < 0.5] = 0

    # ------------------------
    # pick two fitting patches
    # ------------------------

    patch_size = 80
    padding_size = 10
    pairs = find_promising_patch_pairs(feature_map, patch_size=patch_size, stride=16, num_pairs=439)

    patch_slice, window_slice, _ = pairs[0]

    feature_patch = np.pad(feature_map[patch_slice], padding_size)
    window_slice = pad_slices(window_slice, padding=padding_size, assert_shape=feature_map.shape)
    feature_window = feature_map[window_slice]

    assert feature_patch.shape == feature_window.shape

    centroids, intervals = get_n_equidistant_angles_and_intervals(4)

    # results = estimate_transform_from_correspondences(feature_patch, feature_window, 20, centroids,
    #                                                   intervals, verbose=True)
    # results = estimate_dense_displacements_from_memberships(feature_patch, feature_window, 20,
    #                                                         centroids, intervals, smooth=0, verbose=True)
    results = estimate_transform_by_minimizing_energy(feature_patch, feature_window, 20, centroids,
                                                      intervals, smooth=0, verbose=True)

    gif = GifExporter()
    errors = [r.error for r in results]
    energies = [r.energy for r in results]
    for i, result in enumerate(results):
        _, axs = plt.subplots(2, 3, figsize=(9, 6))
        warped = apply_transform(feature_patch, result.stacked_transform)
        plot_diff(warped, feature_window, axs=axs[0])
        axs[1, 2].plot(errors[:i + 1])
        axs[1, 2].set_xlim(0, len(results) - 1)
        axs[1, 2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[1, 1].plot(energies[:i + 1])
        axs[1, 1].set_xlim(0, len(results) - 1)
        axs[1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        gif.add_current_fig()
        plt.close()

    gif.save_gif("data/plot/test_plot.gif", duration=0.5)
