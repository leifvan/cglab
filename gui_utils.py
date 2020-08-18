import pickle
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Sequence, Union

import attr
import imageio
import matplotlib.pyplot as plt
import numpy as np
import skimage
import streamlit as st

from distance_transform import get_binary_assignments_from_centroids, get_binary_assignments_from_gabor, \
    get_memberships_from_centroids, get_memberships_from_gabor
from gradient_directions import get_n_equidistant_angles_and_intervals, get_main_gradient_angles_and_intervals
from methods import estimate_linear_transform, estimate_dense_displacements, apply_transform
from gui_config import RunConfiguration, RUNS_DIRECTORY, CONFIG_SUFFIX, RunResult
import gui_config as conf
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


def load_and_preprocess_feature_map(feature_map_path, downscale_factor):
    feature_map = imageio.imread(feature_map_path).astype(np.float32)

    if len(feature_map.shape) == 3:
        feature_map = np.mean(feature_map, axis=2)

    feature_map = skimage.transform.downscale_local_mean(feature_map, (downscale_factor,
                                                                       downscale_factor))
    feature_map /= feature_map.max()
    feature_map[feature_map > 0.5] = 1
    feature_map[feature_map < 0.5] = 0
    return feature_map


def get_padded_moving_and_static(feature_map, moving_slices, static_slices):
    padded_static_slice = pad_slices(static_slices, padding=conf.PADDING_SIZE,
                                     assert_shape=feature_map.shape)
    moving = np.pad(feature_map[moving_slices], conf.PADDING_SIZE)
    static = feature_map[padded_static_slice]
    return moving, static


def run_config(config: RunConfiguration, pbar):
    feature_map = load_and_preprocess_feature_map(conf.FEATURE_MAP_DIR / config.feature_map_path,
                                                  config.downscale_factor)
    moving, static = get_padded_moving_and_static(feature_map, config.moving_slices, config.static_slices)

    # FIXME this is copy-pasted from gui.py
    if config.centroid_method == conf.CentroidMethod.EQUIDISTANT:
        centroids, intervals = get_n_equidistant_angles_and_intervals(config.num_centroids)
    elif config.centroid_method == conf.CentroidMethod.HISTOGRAM_CLUSTERING:
        centroids, intervals = get_main_gradient_angles_and_intervals(static, config.kde_rho)

    assignment_fn = None

    if config.assignment_type == 'binary' and config.filter_method == 'Farid derivative filter':
        assignment_fn = get_binary_assignments_from_centroids
    elif config.assignment_type == 'binary' and config.filter_method == 'Gabor filter':
        assignment_fn = partial(get_binary_assignments_from_gabor, sigma=config.gabor_filter_sigma)
    elif config.assignment_type == 'memberships' and config.filter_method == 'Farid derivative filter':
        assignment_fn = get_memberships_from_centroids
    elif config.assignment_type == 'memberships' and config.filter_method == 'Gabor filter':
        assignment_fn = partial(get_memberships_from_gabor, sigma=config.gabor_filter_sigma)

    common_params = dict(moving=moving, static=static, n_iter=config.num_iterations,
                         centroids=centroids, intervals=intervals, progress_bar=pbar,
                         assignments_fn=assignment_fn,
                         weight_correspondence_angles=config.weight_correspondence_angles)

    estimate_fn = None
    if config.transform_type == 'linear transform':
        estimate_fn = partial(estimate_linear_transform, reg_factor=config.l2_regularization_factor,
                              ttype=config.linear_transform_type)
    elif config.transform_type == 'dense displacement':
        estimate_fn = partial(estimate_dense_displacements, smooth=config.smoothness,
                              rbf_type=config.rbf_type, reduce_coeffs=config.num_dct_coeffs)

    results = estimate_fn(**common_params)

    if results is not None:
        config.save()
        result_obj = RunResult(moving=moving.copy(), static=static.copy(), centroids=centroids.copy(),
                               intervals=intervals.copy(), results=results,
                               warped_moving=[apply_transform(moving, r.stacked_transform) for r in results])
        config.save_results(result_obj)

    return results

