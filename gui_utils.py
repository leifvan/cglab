import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import attr
import streamlit as st
from typing import List

from methods import TransformResult


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
    file_path: str = attr.ib(default=None, eq=False)
    patch_position: int = attr.ib(default=None)
    centroid_method: str = attr.ib(default=None)
    num_centroids: int = attr.ib(default=None)
    kde_rho: float = attr.ib(default=None)
    assignment_type: str = attr.ib(default=None)
    transform_type: str = attr.ib(default=None)
    smoothness: int = attr.ib(default=None)
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

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as config_file:
            return pickle.load(config_file)

    def save(self):
        with open(self.file_path, 'wb') as config_file:
            pickle.dump(self, config_file)

    def load_results(self) -> RunResult:
        with open(self.file_path.replace(CONFIG_SUFFIX, RESULTS_SUFFIX), 'rb') as results_file:
            return pickle.load(results_file)

    def save_results(self, results):
        with open(self.file_path.replace(CONFIG_SUFFIX, RESULTS_SUFFIX), 'wb') as results_file:
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


RUNS_DIRECTORY = "data/runs"
CONFIG_SUFFIX = ".config"
RESULTS_SUFFIX = ".results"


def load_previous_configs():
    config_paths = [p for p in os.listdir(RUNS_DIRECTORY) if p.endswith(CONFIG_SUFFIX)]
    return [RunConfiguration.load(os.path.join(RUNS_DIRECTORY, p)) for p in config_paths]