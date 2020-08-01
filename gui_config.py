from pathlib import Path
from gui_utils import ParamDescriptor, ValueIterEnum, ParamType, VisType

FEATURE_MAP_DIR = Path("data/feature_maps")

PATCH_SIZE = 80
PADDING_SIZE = 10
DOWNSCALE_FACTOR = 4
NUM_PATCH_PAIRS = 1000
INITIAL_PATCH_PAIR = 250
PATCH_STRIDE = 16

PATCH_POSITION_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=1, max_value=1000,
                                            value=250, step=1, vis_type=VisType.NUMBER_INPUT)


class FilterMethod(ValueIterEnum):
    FARID_DERIVATIVE = 'Farid derivative filter'
    GABOR = 'Gabor filter'


FILTER_METHOD_DESCRIPTOR = ParamDescriptor(ParamType.CATEGORICAL, options=FilterMethod.values())

GABOR_FILTER_SIGMA_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=1., max_value=4.,
                                                value=2., step=0.5)


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


class TransformType(ValueIterEnum):
    LINEAR = 'linear transform'
    DENSE = 'dense displacement'


TRANSFORM_TYPE_DESCRIPTOR = ParamDescriptor(ParamType.CATEGORICAL, options=TransformType.values())

SMOOTHNESS_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=0, max_value=20000,
                                        value=2000, step=100)

L2_REGULARIZATION_FACTOR_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=0,
                                                      max_value=20000, value=2000, step=100)

NUM_DCT_COEFFS_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=0, max_value=100, value=0,
                                            step=1)

NUM_ITERATIONS_DESCRIPTOR = ParamDescriptor(ParamType.INTERVAL, min_value=1, max_value=200,
                                            value=20, step=1, vis_type=VisType.NUMBER_INPUT)
