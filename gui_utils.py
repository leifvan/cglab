import pickle
from enum import Enum
from functools import partial
from pathlib import Path
import random
from typing import Sequence, Union

import attr
import imageio
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import skimage.morphology
import streamlit as st
import warnings

from distance_transform import get_binary_assignments_from_centroids, get_binary_assignments_from_gabor, \
    get_memberships_from_centroids, get_memberships_from_gabor
from gradient_directions import get_n_equidistant_angles_and_intervals, get_main_gradient_angles_and_intervals, \
    get_gradients_in_polar_coords, wrapped_cauchy_kernel_density
from methods import estimate_linear_transform, estimate_dense_displacements, apply_transform
from gui_config import RunConfiguration, RUNS_DIRECTORY, CONFIG_SUFFIX, RunResult
import gui_config as conf
from patches import find_promising_patch_pairs
from utils import pad_slices


class StreamlitProgressWrapper:
    def __init__(self, total, parent=st):
        self.total = total
        self.n = 0
        self.label = parent.text(body=f"0 / {total}")
        self.pbar = parent.progress(0.)
        self.postfix = ""

    def _update_label(self):
        self.label.text(f"{self.n} / {self.total} | {self.postfix}")

    def update(self, delta):
        self.n += delta
        self._update_label()
        if self.n <= self.total:
            self.pbar.progress(self.n / self.total)
        else:
            print("Warning: progress bar >100%")

    def set_postfix(self, postfix):
        self.postfix = postfix
        self._update_label()


def figure_to_image():
    canvas = plt.gcf().canvas
    canvas.draw()
    buf = canvas.buffer_rgba()
    plt.close()
    return np.asarray(buf)


def load_previous_configs() -> Sequence[RunConfiguration]:
    config_paths = [p for p in RUNS_DIRECTORY.glob(f"*{CONFIG_SUFFIX}")]
    return [RunConfiguration.load(p) for p in config_paths]


def angle_to_degrees(centroids):
    return [f"{-(c / np.pi * 180 + 180) % 360:.0f}°" for c in centroids]


def get_kde_scores(patch, kde_rho):
    # TODO this is just copy-and-pasted from the code in gradient_directions.py
    angles, magnitudes = get_gradients_in_polar_coords(patch)

    # flatten
    angles = np.ravel(angles)
    magnitudes = np.ravel(magnitudes)

    # select only pixels where magnitude does not vanish
    indices = np.argwhere(~np.isclose(magnitudes, 0))[:, 0]

    # for very dense feature maps it might make sense to sample points
    indices = random.sample(list(indices), k=min(len(indices), 2000))

    angles = angles[indices]
    magnitudes = magnitudes[indices]

    sample_points = np.linspace(-np.pi, np.pi, 360)
    scores = wrapped_cauchy_kernel_density(theta=sample_points[:, None],
                                           samples=angles[:, None],
                                           weights=magnitudes,
                                           rho=kde_rho)
    return sample_points, scores


def load_and_preprocess_feature_map(feature_map_path, downscale_factor):
    feature_map = imageio.imread(feature_map_path).astype(np.float32)

    if len(feature_map.shape) == 3:
        feature_map = np.mean(feature_map, axis=2)

    feature_map = skimage.transform.downscale_local_mean(feature_map, (downscale_factor,
                                                                       downscale_factor))
    feature_map /= feature_map.max()
    feature_map[feature_map > 0.5] = 1
    feature_map[feature_map < 0.5] = 0
    #feature_map = skimage.morphology.thin(feature_map).astype(np.float)
    #selem = skimage.morphology.disk(radius=9-2*downscale_factor)
    #feature_map = skimage.morphology.binary_erosion(feature_map,selem).astype(np.float)
    return feature_map


def get_padded_moving_and_static(feature_map, moving_slices, static_slices):
    padded_static_slice = pad_slices(static_slices, padding=conf.PADDING_SIZE,
                                     assert_shape=feature_map.shape)
    if feature_map.ndim == 2:
        moving = np.pad(feature_map[moving_slices], conf.PADDING_SIZE)
    elif feature_map.ndim == 3:
        ps = conf.PADDING_SIZE
        moving = np.pad(feature_map[moving_slices], ((ps,ps),(ps,ps),(0,0)))
    else:
        raise ValueError("feature map must be 2D or 3D.")

    static = feature_map[padded_static_slice]
    return moving, static


def get_slices_from_patch_position(feature_map, downscale_factor, patch_position):
    ppp = find_promising_patch_pairs(feature_map, conf.PATCH_SIZE,
                                     conf.PATCH_STRIDE(downscale_factor),
                                     conf.PADDING_SIZE, num_pairs=conf.NUM_PATCH_PAIRS)
    moving_slices, static_slices, _ = ppp[-patch_position]
    return moving_slices, static_slices


def run_config(config: RunConfiguration, pbar):
    feature_map = load_and_preprocess_feature_map(conf.FEATURE_MAP_DIR / config.feature_map_path,
                                                  config.downscale_factor)
    if config.moving_slices is None or config.static_slices is None:
        warnings.warn("No explicit patch slices are given. run_config will use "
                      "find_promising_patch_pairs with the currently configured patch size, "
                      "stride and padding size, which is slow and may lead to inconsistent "
                      "results. Consider passing a config with explicit values for 'moving_slices' "
                      "and 'static_slices'.")
        moving_slices, static_slices = get_slices_from_patch_position(feature_map,
                                                                      config.downscale_factor,
                                                                      config.patch_position)
    else:
        moving_slices, static_slices = config.moving_slices, config.static_slices

    moving, static = get_padded_moving_and_static(feature_map, moving_slices, static_slices)

    # FIXME this is copy-pasted from gui.py
    if config.centroid_method == conf.CentroidMethod.EQUIDISTANT:
        centroids, intervals = get_n_equidistant_angles_and_intervals(config.num_centroids)
    elif config.centroid_method == conf.CentroidMethod.HISTOGRAM_CLUSTERING:
        centroids, intervals = get_main_gradient_angles_and_intervals(static, config.kde_rho)
    else:
        raise ValueError(f"Invalid value '{config.centroid_method}' centroid_method.")

    assignment_fn = None

    # FIXME replace strings with enums

    if config.assignment_type == conf.AssignmentType.BINARY:
        if config.filter_method == conf.FilterMethod.FARID_DERIVATIVE:
            assignment_fn = get_binary_assignments_from_centroids
        elif config.filter_method == conf.FilterMethod.GABOR:
            assignment_fn = partial(get_binary_assignments_from_gabor,
                                    sigma=config.gabor_filter_sigma,
                                    threshold=config.response_cutoff_threshold)

    elif config.assignment_type == conf.AssignmentType.MEMBERSHIPS:
        if config.filter_method == conf.FilterMethod.FARID_DERIVATIVE:
            assignment_fn = get_memberships_from_centroids
        elif config.filter_method == conf.FilterMethod.GABOR:
            assignment_fn = partial(get_memberships_from_gabor,
                                    sigma=config.gabor_filter_sigma,
                                    threshold=config.response_cutoff_threshold)

    if assignment_fn is None:
        raise ValueError(f"No valid assignment function found for "
                         f"assignment_type='{config.assignment_type}' and "
                         f"filter_method='{config.filter_method}'.")

    common_params = dict(moving=moving, static=static, n_iter=config.num_iterations,
                         centroids=centroids, intervals=intervals, progress_bar=pbar,
                         assignments_fn=assignment_fn,
                         weight_correspondence_angles=config.weight_correspondence_angles,
                         reduce_boundary_weights=config.reduce_boundary_weights)

    estimate_fn = None
    if config.transform_type == conf.TransformType.LINEAR:
        estimate_fn = partial(estimate_linear_transform, reg_factor=config.l2_regularization_factor,
                              ttype=config.linear_transform_type)
    elif config.transform_type == conf.TransformType.DENSE:
        estimate_fn = partial(estimate_dense_displacements, smooth=config.smoothness,
                              rbf_type=config.rbf_type, reduce_coeffs=config.num_dct_coeffs)

    estimate_results = estimate_fn(**common_params)
    results_obj = None

    if estimate_results is not None:
        # TODO check out if we really make use of all what is saved here
        results_obj = RunResult(moving=moving.copy(), static=static.copy(), centroids=centroids.copy(),
                                intervals=intervals.copy(), results=estimate_results,
                                warped_moving=[apply_transform(moving, r.stacked_transform) for r in
                                               estimate_results])

        if config.file_path is not None:
            config.save()
            config.save_results(results_obj)
        else:
            warnings.warn("The config and results will not be saved as a file, "
                          "because 'file_path' is None.")

    return results_obj
