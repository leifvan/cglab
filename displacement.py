import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib as mpl
from gradient_directions import plot_polar_gradients, plot_gradients_as_arrows
from utils import get_colored_difference_image, angle_to_rgb
from skimage.transform import ProjectiveTransform, estimate_transform, AffineTransform
from skimage.transform._geometric import _center_and_normalize_points
from scipy.sparse.linalg import lsmr


def calculate_dense_displacements(assignments, distances, directions, smooth):
    """
    Calculates a displacement map based on binary assignments using L2-regularized radial basis
    functions.

    :param assignments: An array (n_angles, height, width) of binary maps s.t. assignments[k, i, j]
        is 1 if pixel [i,j] corresponds to angle k (0 otherwise).
    :param distances: An array (n_angles, height, width) of distances to the next edge pixel, i.e.
        distances[k, i, j] is the distance of pixel [i,j] to the next edge pixel with angle k.
    :param directions: An array (n_angles, height, width) of angles to the next edge pixel, i.e.
        directions[k, i, j] is the angle from [i,j] to the next edge pixel with angle k.
    :param smooth: L2-regularization factor for the RBFs.
    :return: An array (2, height, width) of y and x displacements for each pixel, i.e. [:,i,j] is
        the displacement (y,x) of pixel [i,j].
    """
    height, width = distances.shape[1:3]
    yy, xx = np.mgrid[:height, :width]

    feature_gradient_cartesian = np.array([np.sin(directions), np.cos(directions)])
    feature_gradient_cartesian *= distances * assignments
    feature_gradient_cartesian = feature_gradient_cartesian.sum(axis=1)

    all_locations = np.argwhere(np.logical_or.reduce(assignments, axis=0))
    fy, fx = all_locations[:, 0], all_locations[:, 1]

    transposed_coords = np.transpose(feature_gradient_cartesian[:, fy, fx])
    interpolator = scipy.interpolate.Rbf(fy, fx, transposed_coords, function='linear',
                                         smooth=smooth, mode='N-D')
    interpolated = interpolator(yy, xx)
    return np.moveaxis(interpolated, 2, 0)


def plot_correspondences(moving, static, centroids, assignments, distances, directions, ax=None):
    ax = ax or plt.gca()
    assert np.all(assignments <= 1)
    aa, yy, xx = np.nonzero(assignments)
    angles = np.array([np.sin(directions[aa, yy, xx]), np.cos(directions[aa, yy, xx])])
    uu, vv = angles * distances[aa, yy, xx]
    colors = angle_to_rgb(centroids[aa], with_alpha=True)  # hsv((centroids[aa] + np.pi) / 2 / np.pi)
    colors[:, 3] = assignments[aa, yy, xx] * 0.5
    ax.imshow(get_colored_difference_image(moving, static))
    ax.quiver(xx, yy, -vv, uu, angles='xy', scale_units='xy', scale=1,
              color=colors)


def get_energy(memberships, distances):
    weighted_distances = memberships * distances
    return weighted_distances.sum() / memberships.sum()


def estimate_projective_transform(src, dst, weights=None):
    src_matrix, src = _center_and_normalize_points(src)
    dst_matrix, dst = _center_and_normalize_points(dst)
    n = len(src)
    a = np.zeros((2 * n, 8))
    b = np.concatenate([dst[:, 0], dst[:, 1]])

    if weights is None:
        weights = np.ones(n)

    sqrt_weights = np.sqrt(weights)

    b[:n] *= sqrt_weights
    b[n:] *= sqrt_weights

    a[:n, 0] = src[:, 0]
    a[:n, 1] = src[:, 1]
    a[:n, 2] = 1
    a[:n, 6] = -src[:, 0] * dst[:, 0]
    a[:n, 7] = -src[:, 1] * dst[:, 0]
    a[:n] *= sqrt_weights[..., None]

    a[n:, 3] = src[:, 0]
    a[n:, 4] = src[:, 1]
    a[n:, 5] = 1
    a[n:, 6] = -src[:, 0] * dst[:, 1]
    a[n:, 7] = -src[:, 1] * dst[:, 1]
    a[n:] *= sqrt_weights[..., None]

    # damp is the lambda of Tikhonov regularization
    x = lsmr(a, b, damp=0)[0]

    mat = np.array([*x, 1]).reshape((3, 3))
    mat_transformed = np.linalg.inv(dst_matrix) @ mat @ src_matrix
    return ProjectiveTransform(matrix=mat_transformed)


# def estimate_transform_from_binary_assignments(assignments, distances, directions):
#     assert np.all((assignments == 0) | (assignments == 1))
#     aa, yy, xx = np.nonzero(assignments)
#     angles = np.array([np.sin(directions[aa, yy, xx]), np.cos(directions[aa, yy, xx])])
#     uu, vv = angles * distances[aa, yy, xx]
#     src = np.stack([xx, yy], axis=1)
#     dst = np.stack([xx + vv, yy + uu], axis=1)
#     # src = np.stack([yy, xx], axis=1)
#     # dst = np.stack([yy + vv, xx + uu], axis=1)
#     #return estimate_affine_transform(src, dst)
#     return estimate_projective_transform(src, dst)


def estimate_transform_from_memberships(memberships, distances, directions):
    aa, yy, xx = np.nonzero(memberships)
    angles = np.array([np.sin(directions[aa, yy, xx]), np.cos(directions[aa, yy, xx])])
    uu, vv = angles * distances[aa, yy, xx]
    src = np.stack([xx, yy], axis=1)
    dst = np.stack([xx + vv, yy + uu], axis=1)
    return estimate_projective_transform(src, dst, weights=memberships[aa, yy, xx])
