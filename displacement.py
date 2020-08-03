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

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from utils import get_colored_difference_image, angle_to_rgb
from skimage.transform import ProjectiveTransform, warp
from skimage.transform._geometric import _center_and_normalize_points
from scipy.sparse.linalg import lsmr


# TODO improve docstring for this function
def calculate_dense_displacements(memberships, distances, directions, smooth):
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
    :return: An array (2, height, width) of y and x displacements for each pixel, i.e. [:,i,j] is
        the displacement (y,x) of pixel [i,j].
    """
    height, width = distances.shape[1:3]
    yy, xx = np.mgrid[:height, :width]

    # FIXME proper least squares weighting
    feature_gradient_cartesian = np.array([np.sin(directions), np.cos(directions)])
    feature_gradient_cartesian *= distances * memberships
    feature_gradient_cartesian = feature_gradient_cartesian.sum(axis=1)

    all_locations = np.argwhere(np.logical_or.reduce(memberships, axis=0))
    fy, fx = all_locations[:, 0], all_locations[:, 1]

    transposed_coords = np.transpose(feature_gradient_cartesian[:, fy, fx])
    interpolator = scipy.interpolate.Rbf(fy, fx, transposed_coords, function='linear',
                                         smooth=smooth, mode='N-D')
    interpolated = interpolator(yy, xx)
    return np.moveaxis(interpolated, 2, 0)


def plot_correspondences(moving, static, centroids, memberships, distances, directions, ax=None):
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
    aa, yy, xx = np.nonzero(memberships)
    angles = np.array([np.sin(directions[aa, yy, xx]), np.cos(directions[aa, yy, xx])])
    uu, vv = angles * distances[aa, yy, xx]
    colors = angle_to_rgb(centroids[aa], with_alpha=True)  # hsv((centroids[aa] + np.pi) / 2 / np.pi)
    colors[:, 3] = memberships[aa, yy, xx] * 0.5
    ax.imshow(get_colored_difference_image(moving, static))
    ax.quiver(xx, yy, -vv, uu, angles='xy', scale_units='xy', scale=1,
              color=colors)


def get_energy(memberships, distances):
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


def estimate_projective_transform(src, dst, weights=None, reg_factor=0.):
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
    src_matrix, src = _center_and_normalize_points(src)
    dst_matrix, dst = _center_and_normalize_points(dst)
    n = len(src)
    a = np.zeros((2 * n, 8))
    b = np.concatenate([dst[:, 0] - src[:, 0], dst[:, 1] - src[:, 1]])

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
    x = lsmr(a, b, damp=reg_factor)[0]

    mat = np.array([*x, 1]).reshape((3, 3))
    mat[0,0] += 1
    mat[1,1] += 1
    mat_transformed = np.linalg.inv(dst_matrix) @ mat @ src_matrix
    return ProjectiveTransform(matrix=mat_transformed)


def plot_projective_transform(transform, ax=None):
    if ax is None:
        ax = plt.gca()

    unwarped = np.zeros((200, 200))
    unwarped[50:150, 50:150] = 1
    warped = warp(unwarped, transform)
    ax.imshow(0.8*warped-0.2*unwarped, cmap='bone_r', vmin=0, vmax=1)


def estimate_transform_from_memberships(memberships, distances, directions, reg_factor=0.):
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
    aa, yy, xx = np.nonzero(memberships)
    angles = np.array([np.sin(directions[aa, yy, xx]), np.cos(directions[aa, yy, xx])])
    uu, vv = angles * distances[aa, yy, xx]
    src = np.stack([xx, yy], axis=1)
    dst = np.stack([xx + vv, yy + uu], axis=1)
    return estimate_projective_transform(src, dst, weights=memberships[aa, yy, xx],
                                         reg_factor=reg_factor)
