from collections import OrderedDict
from pathlib import Path

from gui_utils import ParamDescriptor, ValueIterEnum, ParamType, VisType, PartialRunConfiguration

FEATURE_MAP_DIR = Path("data/feature_maps")
TEXTURE_DIR = Path("data/textures")

PATCH_SIZE = 80
PADDING_SIZE = 10
NUM_PATCH_PAIRS = 1000
INITIAL_PATCH_PAIR = 250
# TODO reparameterize this somehow:
# PATCH_STRIDE = 16

DOWNSCALE_FACTOR_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=1, max_value=4,
                                              value=4, step=1, vis_type=VisType.NUMBER_INPUT)

PATCH_POSITION_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=1, max_value=1000,
                                            value=250, step=1, vis_type=VisType.NUMBER_INPUT)


class FilterMethod(ValueIterEnum):
    FARID_DERIVATIVE = 'Farid derivative filter'
    GABOR = 'Gabor filter'


FILTER_METHOD_DESCRIPTOR = ParamDescriptor(ParamType.CATEGORICAL, options=FilterMethod.values())

GABOR_FILTER_SIGMA_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=0.1, max_value=2.,
                                                value=0.5, step=0.1)

RESPONSE_CUTOFF_THRESHOLD = ParamDescriptor(ParamType.INTERVAL, min_value=-1., max_value=0.,
                                            value=-1/4., step=1/32, exponential=True)


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


class TransformType(ValueIterEnum):
    LINEAR = 'linear transform'
    DENSE = 'dense displacement'


TRANSFORM_TYPE_DESCRIPTOR = ParamDescriptor(ParamType.CATEGORICAL, options=TransformType.values())


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

NUM_DCT_COEFFS_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=0, max_value=100, value=0,
                                            step=1)

NUM_ITERATIONS_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=1, max_value=200,
                                            value=20, step=1, vis_type=VisType.NUMBER_INPUT)

PARAM_DESCRIPTOR_MAP = OrderedDict(downscale_factor=DOWNSCALE_FACTOR_DESCRIPTOR,
                                   patch_position=PATCH_POSITION_DESCRIPTOR,
                                   filter_method=FILTER_METHOD_DESCRIPTOR,
                                   gabor_filter_sigma=GABOR_FILTER_SIGMA_DESCRIPTOR,
                                   response_cutoff_threshold=RESPONSE_CUTOFF_THRESHOLD,
                                   centroid_method=CENTROID_METHOD_DESCRIPTOR,
                                   num_centroids=NUM_CENTROIDS_DESCRIPTOR,
                                   kde_rho=KDE_RHO_DESCRIPTOR,
                                   assignment_type=ASSIGNMENT_TYPE_DESCRIPTOR,
                                   transform_type=TRANSFORM_TYPE_DESCRIPTOR,
                                   smoothness=SMOOTHNESS_DESCRIPTOR,
                                   l2_regularization_factor=L2_REGULARIZATION_FACTOR_DESCRIPTOR,
                                   num_dct_coeffs=NUM_DCT_COEFFS_DESCRIPTOR,
                                   num_iterations=NUM_ITERATIONS_DESCRIPTOR)

DEFAULT_CONFIG = PartialRunConfiguration(**{param: desc.value for param, desc in PARAM_DESCRIPTOR_MAP.items()})
