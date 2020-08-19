import pickle
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Sequence, Union, Tuple

import attr
import numpy as np
import streamlit as st

from utils import TransformResult

FEATURE_MAP_DIR = Path("data/feature_maps")
TEXTURE_DIR = Path("data/textures")
RUNS_DIRECTORY = Path("data/runs")
CONFIG_SUFFIX = ".config"
RESULTS_SUFFIX = ".results"

PATCH_SIZE = 80
PADDING_SIZE = 10
NUM_PATCH_PAIRS = 1000
INITIAL_PATCH_PAIR = 250


def PATCH_STRIDE(ds):
    return 64 // ds


@attr.s
class RunResult:
    moving: np.ndarray = attr.ib()
    static: np.ndarray = attr.ib()

    centroids: np.ndarray = attr.ib()
    intervals: np.ndarray = attr.ib()

    results: Sequence[TransformResult] = attr.ib()
    warped_moving: list = attr.ib()


# TODO enforce correct values with enums
@attr.s
class PartialRunConfiguration:
    feature_map_path: str = attr.ib(default=None)
    file_path: Path = attr.ib(default=None, eq=False)
    downscale_factor: int = attr.ib(default=None)
    patch_position_type: str = attr.ib(default=None)
    patch_position: int = attr.ib(default=None)  # FIXME about to be deprecated
    moving_slices: Sequence[slice] = attr.ib(default=None)
    static_slices: Sequence[slice] = attr.ib(default=None)
    filter_method: str = attr.ib(default=None)
    gabor_filter_sigma: float = attr.ib(default=None)
    response_cutoff_threshold: float = attr.ib(default=None)
    centroid_method: str = attr.ib(default=None)
    num_centroids: int = attr.ib(default=None)
    kde_rho: float = attr.ib(default=None)
    assignment_type: str = attr.ib(default=None)
    weight_correspondence_angles: bool = attr.ib(default=None)
    reduce_boundary_weights: bool = attr.ib(default=None)
    transform_type: str = attr.ib(default=None)
    linear_transform_type: str = attr.ib(default=None)
    rbf_type: str = attr.ib(default=None)
    smoothness: int = attr.ib(default=None)
    l2_regularization_factor: float = attr.ib(default=None)
    num_dct_coeffs: int = attr.ib(default=None)
    num_iterations: int = attr.ib(default=None)
    is_favorite: bool = attr.ib(default=False, eq=False)


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

    @property
    def name(self):
        return self.file_path.stem

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


class ParamType(Enum):
    INTERVAL = "interval"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class VisType(Enum):
    DEFAULT = "default"

    # for interval-valued params (default: slider)
    SLIDER = "slider"
    NUMBER_INPUT = "number_input"

    # for categorical params (default: radio)
    RADIO = "radio"
    SELECTBOX = "selectbox"
    MULTISELECT = "multiselect"

    # for boolean params (default: checkbox)
    CHECKBOX = "checkbox"


def _validate_param_type(instance, attribute, value):
    return instance.param_type == attribute.metadata['param_type'] or value is None


_interval_meta = dict(param_type=ParamType.INTERVAL)
_interval_meta_no_param = dict(param_type=ParamType.INTERVAL)
_categorical_meta = dict(param_type=ParamType.CATEGORICAL)


@attr.s
class ParamDescriptor:
    param_type: Union[ParamType] = attr.ib()
    vis_type: VisType = attr.ib(default=VisType.DEFAULT)

    # interval
    min_value: float = attr.ib(default=None, validator=_validate_param_type, metadata=_interval_meta)
    max_value: float = attr.ib(default=None, validator=_validate_param_type, metadata=_interval_meta)
    value: float = attr.ib(default=None, validator=_validate_param_type, metadata=_interval_meta)
    step: float = attr.ib(default=None, validator=_validate_param_type, metadata=_interval_meta)
    exponential: bool = attr.ib(default=False)

    # categorical
    options: Sequence[str] = attr.ib(default=None, validator=_validate_param_type, metadata=_categorical_meta)
    index: int = attr.ib(default=None, validator=_validate_param_type, metadata=_categorical_meta)
    default: Sequence[str] = attr.ib(default=None, validator=_validate_param_type, metadata=_categorical_meta)

    @vis_type.validator
    def _validate_vis_type(self, attribute, value):
        if value == VisType.DEFAULT:
            return True

        if self.param_type == ParamType.INTERVAL:
            return value in (VisType.SLIDER, VisType.NUMBER_INPUT)
        elif self.param_type == ParamType.CATEGORICAL:
            return value in (VisType.RADIO, VisType.SELECTBOX)
        elif self.param_type == ParamType.BOOLEAN:
            return value == VisType.CHECKBOX

    @property
    def param_dict(self):
        return attr.asdict(self, filter=lambda a, v: a.metadata and a.metadata['param_type'] == self.param_type)

    def validate_result(self, result):
        if self.param_type == ParamType.INTERVAL:
            return self.min_value <= result <= self.max_value
        elif self.param_type == ParamType.CATEGORICAL:
            return result in self.options
        return True


def make_st_widget(descriptor: ParamDescriptor, label: str, target=st.sidebar, value=None,
                   returns_iterable=False):
    params = {k: v for k, v in descriptor.param_dict.items() if v is not None}
    factory = None

    if descriptor.param_type == ParamType.INTERVAL:
        if descriptor.exponential:
            params["format"] = "1e%f"
        if descriptor.vis_type in (VisType.DEFAULT, VisType.SLIDER):
            factory = target.slider
        elif descriptor.vis_type == VisType.NUMBER_INPUT:
            factory = target.number_input
    elif descriptor.param_type == ParamType.CATEGORICAL:
        if descriptor.vis_type in (VisType.DEFAULT, VisType.RADIO):
            factory = target.radio
        elif descriptor.vis_type == VisType.SELECTBOX:
            factory = target.selectbox
        elif descriptor.vis_type == VisType.MULTISELECT:
            factory = target.multiselect
            returns_iterable = True
    elif descriptor.param_type == ParamType.BOOLEAN:
        factory = target.checkbox
    else:
        raise AttributeError(descriptor)

    if value is not None:
        if descriptor.param_type == ParamType.INTERVAL:
            params['value'] = value
        elif descriptor.param_type == ParamType.CATEGORICAL:
            params['index'] = descriptor.options.index(value)

    value = factory(label, **params)
    if returns_iterable:
        assert not descriptor.exponential  # not implemented
        assert all(descriptor.validate_result(v) for v in value)
    else:
        assert descriptor.validate_result(value)

    if descriptor.exponential:
        return 10 ** value
    else:
        return value


class ValueIterEnum(str, Enum):
    @classmethod
    def values(cls):
        return [c.value for c in cls]


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                              DESCRIPTORS
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


DOWNSCALE_FACTOR_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=1, max_value=4,
                                              value=4, step=1, vis_type=VisType.NUMBER_INPUT)


class PatchPositionType(ValueIterEnum):
    BEST = 'best'
    WORST = 'worst'


PATCH_POSITION_TYPE_DESCRIPTOR = ParamDescriptor(ParamType.CATEGORICAL,
                                                 options=PatchPositionType.values())

PATCH_POSITION_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=1, max_value=1000,
                                            value=250, step=1, vis_type=VisType.NUMBER_INPUT)


class FilterMethod(ValueIterEnum):
    FARID_DERIVATIVE = 'farid'
    GABOR = 'gabor'


FILTER_METHOD_DESCRIPTOR = ParamDescriptor(ParamType.CATEGORICAL, options=FilterMethod.values())

GABOR_FILTER_SIGMA_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=0.1, max_value=2.,
                                                value=0.5, step=0.1)

RESPONSE_CUTOFF_THRESHOLD = ParamDescriptor(ParamType.INTERVAL, min_value=-1., max_value=0.,
                                            value=-1 / 4., step=1 / 32, exponential=True)


class CentroidMethod(ValueIterEnum):
    EQUIDISTANT = 'equidistant'
    HISTOGRAM_CLUSTERING = 'histogram clustering'


CENTROID_METHOD_DESCRIPTOR = ParamDescriptor(ParamType.CATEGORICAL, options=CentroidMethod.values())

NUM_CENTROIDS_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=1, max_value=32, value=8,
                                           step=1, vis_type=VisType.NUMBER_INPUT)

KDE_RHO_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=0., max_value=1., value=0.8)


class AssignmentType(ValueIterEnum):
    BINARY = 'binary'
    MEMBERSHIPS = 'memberships'


ASSIGNMENT_TYPE_DESCRIPTOR = ParamDescriptor(ParamType.CATEGORICAL, options=AssignmentType.values())

WEIGHT_CORRESPONDENCE_ANGLES_DESCRIPTOR = ParamDescriptor(ParamType.BOOLEAN)

REDUCE_BOUNDARY_WEIGHTS_DESCRIPTOR = ParamDescriptor(ParamType.BOOLEAN)


class TransformType(ValueIterEnum):
    LINEAR = 'linear'
    DENSE = 'dense'


TRANSFORM_TYPE_DESCRIPTOR = ParamDescriptor(ParamType.CATEGORICAL, options=TransformType.values())


class LinearTransformType(ValueIterEnum):
    TRANSPOSITION = 'transposition'
    AFFINE = 'affine'
    PROJECTIVE = 'projective'


LINEAR_TRANSFORM_TYPE_DESCRIPTOR = ParamDescriptor(ParamType.CATEGORICAL, options=LinearTransformType.values())


class RbfType(ValueIterEnum):
    LINEAR = 'linear'
    MULTIQUADRIC = 'multiquadric'
    THIN_PLATE_SPLINES = "thin-plate splines"


RBF_TYPE_DESCRIPTOR = ParamDescriptor(ParamType.CATEGORICAL, options=RbfType.values())

# SMOOTHNESS_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=0, max_value=20000,
#                                         value=2000, step=100)
SMOOTHNESS_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=1., max_value=8.,
                                        value=4., step=0.25, exponential=True)

L2_REGULARIZATION_FACTOR_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=0,
                                                      max_value=20000, value=2000, step=100)

NUM_DCT_COEFFS_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=1,
                                            max_value=PATCH_SIZE + 2 * PADDING_SIZE,
                                            value=PATCH_SIZE + 2 * PADDING_SIZE, step=1)

NUM_ITERATIONS_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=1, max_value=200,
                                            value=20, step=1, vis_type=VisType.NUMBER_INPUT)

PARAM_DESCRIPTOR_MAP = OrderedDict(downscale_factor=DOWNSCALE_FACTOR_DESCRIPTOR,
                                   patch_position_type=PATCH_POSITION_TYPE_DESCRIPTOR,
                                   patch_position=PATCH_POSITION_DESCRIPTOR,
                                   filter_method=FILTER_METHOD_DESCRIPTOR,
                                   gabor_filter_sigma=GABOR_FILTER_SIGMA_DESCRIPTOR,
                                   response_cutoff_threshold=RESPONSE_CUTOFF_THRESHOLD,
                                   centroid_method=CENTROID_METHOD_DESCRIPTOR,
                                   num_centroids=NUM_CENTROIDS_DESCRIPTOR,
                                   kde_rho=KDE_RHO_DESCRIPTOR,
                                   assignment_type=ASSIGNMENT_TYPE_DESCRIPTOR,
                                   weight_correspondence_angles=WEIGHT_CORRESPONDENCE_ANGLES_DESCRIPTOR,
                                   reduce_boundary_weights=REDUCE_BOUNDARY_WEIGHTS_DESCRIPTOR,
                                   transform_type=TRANSFORM_TYPE_DESCRIPTOR,
                                   linear_transform_type=LINEAR_TRANSFORM_TYPE_DESCRIPTOR,
                                   rbf_type=RBF_TYPE_DESCRIPTOR,
                                   smoothness=SMOOTHNESS_DESCRIPTOR,
                                   l2_regularization_factor=L2_REGULARIZATION_FACTOR_DESCRIPTOR,
                                   num_dct_coeffs=NUM_DCT_COEFFS_DESCRIPTOR,
                                   num_iterations=NUM_ITERATIONS_DESCRIPTOR)

DEFAULT_CONFIG = PartialRunConfiguration(**{param: desc.value for param, desc in PARAM_DESCRIPTOR_MAP.items()})
