import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import attr
import streamlit as st


@attr.s(frozen=True)
class RunConfiguration:
    # TODO also save feature map (path)
    patch_position: int = attr.ib(default=None)
    centroid_method: str = attr.ib(default=None)
    num_centroids: int = attr.ib(default=None)
    kde_rho: float = attr.ib(default=None)
    assignment_type: str = attr.ib(default=None)
    transform_type: str = attr.ib(default=None)
    smoothness: int = attr.ib(default=None)
    num_iterations: int = attr.ib(default=None)

    def fulfills(self, proto_config):
        attr_names = attr.fields_dict(RunConfiguration)
        reduced_self = RunConfiguration(**{n: getattr(self, n) for n in attr_names
                                           if getattr(proto_config, n) is not None})
        return reduced_self == proto_config


@attr.s
class RunResult:
    moving: np.ndarray = attr.ib()
    static: np.ndarray = attr.ib()

    centroids: np.ndarray = attr.ib()
    intervals: np.ndarray = attr.ib()

    results: list = attr.ib()
    warped_moving: list = attr.ib()


class StreamlitProgressWrapper:
    def __init__(self, total):
        self.total = total
        self.n = 0
        self.label = st.text(body=f"0 / {total}")
        self.pbar = st.progress(0.)
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
    configs = []
    config_paths = [p for p in os.listdir(RUNS_DIRECTORY) if p.endswith(CONFIG_SUFFIX)]
    for fp in config_paths:
        with open(os.path.join(RUNS_DIRECTORY, fp), 'rb') as config_file:
            configs.append(pickle.load(config_file))
    return configs, config_paths