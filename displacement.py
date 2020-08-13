"""
This module contains functions acting on ``memberships``, ``distances`` and ``directions``. These
are mainly used to retrieve correspondences in the following way:

``memberships`` is a stack of 2D arrays that give a soft-assignment / membership for each pixel of
the *moving image* to the main gradient directions. For example, ``memberships[k, i, j]`` is the
membership of pixel (i,j) in the moving image to the k-th gradient direction.

``distances`` and ``directions`` are also stacks of 2D arrays that give the distance and direction
to the next edge pixel for each pixel in the *static image*. For example, given pixel (i,j) in the
static image, ``distances[k, i, j]`` is the euclidean distance to the closest edge pixel that is
assigned to the k-th main direction and ``directions[k, i, j]`` is the angle of the vector from
(i,j) to the closest edge pixel.

Note that these two arrays require a binary assignment to be generated.
"""

import attr
import matplotlib.pyplot as plt
import numpy as np
from numexpr import evaluate
from scipy.sparse.linalg import lsmr
from scipy.spatial.distance import pdist, squareform, cdist
from skimage.transform import ProjectiveTransform, warp

import gui_config as conf
from utils import get_colored_difference_image, angle_to_rgb


@attr.s(frozen=True)
class Correspondences:
    src: np.ndarray = attr.ib()
    dst: np.ndarray = attr.ib()
    weights: np.ndarray = attr.ib()
    aa: np.ndarray = attr.ib()

    @classmethod
    def from_memberships(cls, memberships, distances, directions, centroids=None):
        aa, yy, xx = np.nonzero(memberships)
        angles_cartesian = np.array([np.sin(directions[aa, yy, xx]), np.cos(directions[aa, yy, xx])])
        uu, vv = angles_cartesian * distances[aa, yy, xx]

        src = np.stack([yy, xx], axis=1)
        dst = np.stack([yy + uu, xx + vv], axis=1)
        weights = memberships[aa, yy, xx].astype(np.float64)

        if centroids is not None:
            centroids_cartesian = np.array([np.sin(centroids), np.cos(centroids)])
            similarity = np.sum(angles_cartesian * centroids_cartesian[:, aa], axis=0)
            weights *= np.clip(similarity, 0., 1.)

        return cls(src=src, dst=dst, weights=weights, aa=aa)

    def get_yxuv(self):
        yy, xx = self.src[:, 0], self.src[:, 1]
        uu, vv = yy - self.dst[:, 0], xx - self.dst[:, 1]
        return yy, xx, uu, vv


def _rbf_linear(r):
    return r


def _rbf_multiquadric(r):
    eps = 0.1
    return evaluate("sqrt((1 / eps *r) ** 2 + 1)")


def _rbf_thin_plate_splines(r):
    return evaluate("r * log(r**r)")


_rbf_type_to_function = {conf.RbfType.LINEAR: _rbf_linear,
                         conf.RbfType.MULTIQUADRIC: _rbf_multiquadric,
                         conf.RbfType.THIN_PLATE_SPLINES: _rbf_thin_plate_splines}


def estimate_dense_warp_field(c: Correspondences, smooth, rbf_type, shape):
    height, width = shape
    yy, xx = np.mgrid[:height, :width]

    offsets = c.dst - c.src
    psi = _rbf_type_to_function[rbf_type]
    location_distances = psi(squareform(pdist(c.src)))
    a_mat = location_distances * c.weights[..., None]
    b_vec = offsets * c.weights[..., None]

    y_weights = lsmr(a_mat, b_vec[:, 0], damp=smooth)[0]
    x_weights = lsmr(a_mat, b_vec[:, 1], damp=smooth)[0]
    ls_weights = np.stack([y_weights, x_weights], axis=1)

    flat_grid = np.stack([yy.ravel(), xx.ravel()], axis=1)
    grid_location_distances = psi(cdist(flat_grid, c.src))
    interpolated = np.transpose(grid_location_distances @ ls_weights)

    return np.reshape(interpolated, (2, height, width))


# TODO improve docstring for this function
def calculate_dense_displacements(memberships, distances, directions, smooth, rbf_type, centroids):
    """
    Calculates a displacement map based on memberships using L2-regularized radial basis
    functions.

    :param memberships: An array (n_angles, height, width) of soft-assignments / memberships s.t.
        memberships[k, i, j] is the membership of pixel [i,j] to angle k.
    :param distances: An array (n_angles, height, width) of distances to the next edge pixel, i.e.
        distances[k, i, j] is the distance of pixel [i,j] to the next edge pixel with angle k.
    :param directions: An array (n_angles, height, width) of angles to the next edge pixel, i.e.
        directions[k, i, j] is the angle from [i,j] to the next edge pixel with angle k.
    :param smooth: L2-regularization factor for the RBFs.
    :param rbf_type: The type of radial basis function to use.
    :return: An array (2, height, width) of y and x displacements for each pixel, i.e. [:,i,j] is
        the displacement (y,x) of pixel [i,j].
    """
    correspondences = Correspondences.from_memberships(memberships, distances, directions, centroids)
    return estimate_dense_warp_field(correspondences, smooth, rbf_type, memberships.shape[1:])


def plot_correspondences(moving, static, centroids, memberships, distances, directions,
                         weight_correspondence_angles, ax=None):
    """
    Plots correspondences for each edge pixel in ``moving`` to its closest pixel in ``centroids``.
    More specifically, each non-null value in ``memberships`` is considered an edge-pixel of a
    main direction. Suppose ``memberships[k, i, j]`` is non-null. Then, ``distances[k, i, j]``
    and ``directions[k, i, j]`` contain the distance and direction from the pixel ``moving[i, j]``
    to the closest pixel in ``static`` with respect to the binary map of ``centroids[k]``. For each
    such correspondence, an arrow is drawn onto the difference image of ``moving`` and ``static``.
    The arrows are colored w.r.t. the angle of the corresponding main direction in ``centroids``.

    :param moving: The 2D moving image of shape (height, width).
    :param static: The 2D static image of shape (height, width).
    :param centroids: An array (n_angles,) of main directions.
    :param memberships: An array (n_angles, height, width) of soft-assignments / memberships s.t.
        memberships[k, i, j] is the membership of pixel [i,j] to angle k.
    :param distances: An array (n_angles, height, width) of distances to the next edge pixel, i.e.
        distances[k, i, j] is the distance of pixel [i,j] to the next edge pixel with angle k.
    :param directions: An array (n_angles, height, width) of angles to the next edge pixel, i.e.
        directions[k, i, j] is the angle from [i,j] to the next edge pixel with angle k.
    :param ax: An optional matplotlib axis to draw the plot onto. If ``None``, the current axis
        will be used.
    """
    ax = ax or plt.gca()
    assert np.all(memberships <= 1)
    centroids_for_weighting = centroids if weight_correspondence_angles else None
    c = Correspondences.from_memberships(memberships, distances, directions, centroids_for_weighting)
    aa = c.aa
    yy, xx, uu, vv = c.get_yxuv()
    colors = angle_to_rgb(centroids[aa], with_alpha=True)
    colors[:, 3] = c.weights * 0.8

    base_difference = get_colored_difference_image(moving, static)

    non_null_membership = memberships.sum(axis=0) > 0
    zero_distance = np.any(distances == 0, axis=0)
    additional_diff_from_filters = get_colored_difference_image(non_null_membership, zero_distance)

    overlaid_differences = base_difference + 0.3 * additional_diff_from_filters
    overlaid_differences /= overlaid_differences.max()
    ax.imshow(overlaid_differences)

    ax.quiver(xx, yy, vv, uu, angles='xy', scale_units='xy', scale=1,
              color=colors)


def get_correspondences_energy(memberships, distances):
    """
    Returns the sum of weighted distances divided by the sum of memberships, i.e. the sum over
    all ``memberships[k,i,j] * distances[k,i,j]`` divided by the sum over all
    ``memberships[k, i, j]``.
    :param memberships: An array (n_angles, height, width) of soft-assignments / memberships s.t.
        memberships[k, i, j] is the membership of pixel [i,j] to angle k.
    :param distances: An array (n_angles, height, width) of distances to the next edge pixel, i.e.
        distances[k, i, j] is the distance of pixel [i,j] to the next edge pixel with angle k.
    :return: The energy value given the memberships and distances.
    """
    weighted_distances = memberships * distances
    return weighted_distances.sum() / memberships.sum()


def estimate_projective_transform(c, reg_factor=0.):
    """
    Estimates an optimal projective transform (in the least-squares sense) given n correspondences
    from ``src`` to ``dst``, i.e. every pair ``(src[i], dst[i])`` is a correspondence. The pairs
    may optionally be weighted, where ``weights[i]`` is the weight of the i-th pair.

    :param src: An array of shape (n, 2) of source points of the n correspondences.
    :param dst: An array of shape (n, 2) of destination points of the n correspondences.
    :param weights: An optional array of shape (n,) of weights for the n correspondences. If
        ``None``, all correspondences are equally weighted.
    :param reg_factor: An optional factor for regularizing the least squares solution. See
        parameter ``damp`` in :func:`scipy.sparse.linalg.lsmr` for details.
    :return: An ``skimage.transform.ProjectiveTransform`` instance that describes an optimal
        projective transform from ``src`` to ``dst`` points.
    """
    src, dst = c.src, c.dst
    n = len(src)
    a = np.zeros((2 * n, 8))
    b = np.concatenate([dst[:, 1] - src[:, 1], dst[:, 0] - src[:, 0]])

    sqrt_weights = np.sqrt(c.weights)

    b[:n] *= sqrt_weights
    b[n:] *= sqrt_weights

    a[:n, 0] = src[:, 1]
    a[:n, 1] = src[:, 0]
    a[:n, 2] = 1
    a[:n, 6] = -src[:, 1] * dst[:, 1]
    a[:n, 7] = -src[:, 0] * dst[:, 1]
    a[:n] *= sqrt_weights[..., None]

    a[n:, 3] = src[:, 1]
    a[n:, 4] = src[:, 0]
    a[n:, 5] = 1
    a[n:, 6] = -src[:, 1] * dst[:, 0]
    a[n:, 7] = -src[:, 0] * dst[:, 0]
    a[n:] *= sqrt_weights[..., None]

    # damp is the lambda of Tikhonov regularization
    x = lsmr(a, b, damp=reg_factor)[0]

    mat = np.array([*x, 0]).reshape((3, 3))
    mat += np.eye(3)
    return ProjectiveTransform(matrix=mat)


def plot_projective_transform(transform, ax=None):
    if ax is None:
        ax = plt.gca()

    unwarped = np.zeros((200, 200))
    unwarped[50:150, 50:150] = 1
    warped = warp(unwarped, transform)
    ax.imshow(0.8 * warped - 0.2 * unwarped, cmap='bone_r', vmin=0, vmax=1)


def estimate_transform_from_memberships(memberships, distances, directions, reg_factor=0., centroids=None):
    """
    Estimates a projective transform based on the correspondences inferred by the given arrays.
    More specifically, each non-null value in ``memberships`` is considered an edge-pixel of a
    main direction. Suppose ``memberships[k, i, j]`` is non-null. Then, ``distances[k, i, j]``
    and ``directions[k, i, j]`` contain the distance and direction from the pixel ``moving[i, j]``
    to the closest pixel in ``static`` with respect to the binary map of ``centroids[k]``. These
    pairs infer a linear system of equations that has an optimal (least-squares) solution.

    If ``memberships`` is non-binary, the values are used to weight the least-squares estimate, i.e.
    higher membership values are given more importance in the least-squares estimate than lower
    ones.

    :param memberships: An array (n_angles, height, width) of soft-assignments / memberships s.t.
        memberships[k, i, j] is the membership of pixel [i,j] to angle k.
    :param distances: An array (n_angles, height, width) of distances to the next edge pixel, i.e.
        distances[k, i, j] is the distance of pixel [i,j] to the next edge pixel with angle k.
    :param directions: An array (n_angles, height, width) of angles to the next edge pixel, i.e.
        directions[k, i, j] is the angle from [i,j] to the next edge pixel with angle k.
    :param reg_factor: An optional factor for regularizing the least squares solution. See
        parameter ``damp`` in :func:`scipy.sparse.linalg.lsmr` for details.
    :return: An ``skimage.transform.ProjectiveTransform`` instance that describes an optimal
        projective transform (in the least-squares sense) of the inferred correspondences.
    """
    correspondences = Correspondences.from_memberships(memberships, distances, directions, centroids)
    return estimate_projective_transform(correspondences, reg_factor=reg_factor)
