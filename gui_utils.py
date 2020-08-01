import pickle
import numpy as np
import matplotlib.pyplot as plt
import attr
import streamlit as st
from typing import List
from pathlib import Path

from methods import TransformResult


@attr.s
class GuiState:
    # TODO currently unused
    feature_map: np.ndarray = attr.ib(default=None)

    patch_pairs: tuple = attr.ib(default=None)
    patch_slice: slice = attr.ib(default=None)
    window_slice: slice = attr.ib(default=None)
    intersection_slice: slice = attr.ib(default=None)

    moving: np.ndarray = attr.ib(default=None)
    static: np.ndarray = attr.ib(default=None)

    centroids: np.ndarray = attr.ib(default=None)
    intervals: np.ndarray = attr.ib(default=None)

    moving_assignments: np.ndarray = attr.ib(default=None)
    moving_memberships: np.ndarray = attr.ib(default=None)
    static_assignments: np.ndarray = attr.ib(default=None)
    static_distances: np.ndarray = attr.ib(default=None)
    static_directions: np.ndarray = attr.ib(default=None)


@attr.s
class RunResult:
    moving: np.ndarray = attr.ib()
    static: np.ndarray = attr.ib()

    centroids: np.ndarray = attr.ib()
    intervals: np.ndarray = attr.ib()

    results: List[TransformResult] = attr.ib()
    warped_moving: list = attr.ib()



@attr.s
class PartialRunConfiguration:
    feature_map_path: str = attr.ib(default=None)
    file_path: Path = attr.ib(default=None, eq=False)
    patch_position: int = attr.ib(default=None)
    filter_method: str = attr.ib(default=None)
    gabor_filter_sigma: float = attr.ib(default=None)
    centroid_method: str = attr.ib(default=None)
    num_centroids: int = attr.ib(default=None)
    kde_rho: float = attr.ib(default=None)
    assignment_type: str = attr.ib(default=None)
    transform_type: str = attr.ib(default=None)
    smoothness: int = attr.ib(default=None)
    l2_regularization_factor: float = attr.ib(default=None)
    num_dct_coeffs: int = attr.ib(default=None)
    num_iterations: int = attr.ib(default=None)


@attr.s(frozen=True)
class RunConfiguration(PartialRunConfiguration):
    _similarity_params = ('patch_position',)

    def fulfills(self, proto_config: 'RunConfiguration'):
        attr_names = attr.fields_dict(RunConfiguration)
        reduced_self = RunConfiguration(**{n: getattr(self, n) for n in attr_names
                                           if getattr(proto_config, n) is not None})
        return reduced_self == proto_config

    def is_similar_to(self, other_config: 'RunConfiguration'):
        return all(getattr(self, sp) == getattr(other_config, sp) for sp in self._similarity_params)

    @property
    def results_path(self):
        return self.file_path.with_suffix(RESULTS_SUFFIX)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as config_file:
            return pickle.load(config_file)

    def save(self):
        with self.file_path.open('wb') as config_file:
            pickle.dump(self, config_file)

    def load_results(self) -> RunResult:
        with self.results_path.open('rb') as results_file:
            return pickle.load(results_file)

    def save_results(self, results):
        with self.results_path.open('wb') as results_file:
            pickle.dump(results, results_file)


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


RUNS_DIRECTORY = Path("data/runs")
CONFIG_SUFFIX = ".config"
RESULTS_SUFFIX = ".results"


def load_previous_configs():
    config_paths = [p for p in RUNS_DIRECTORY.glob(f"*{CONFIG_SUFFIX}")]
    return [RunConfiguration.load(p) for p in config_paths]

def angle_to_degrees(centroids):
    return [f"{-(c / np.pi * 180 + 180) % 360:.0f}°" for c in centroids]