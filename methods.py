import numpy as np
import skimage.transform
from functools import partial
from displacement import get_correspondences_energy, estimate_transform_from_memberships, calculate_dense_displacements
from distance_transform import get_binary_assignments_from_centroids, get_distance_transforms_from_binary_assignments, \
    get_closest_feature_directions_from_binary_assignments, get_memberships_from_centroids, \
    get_binary_assignments_from_gabor
from gradient_directions import get_n_equidistant_angles_and_intervals
from skimage.transform import AffineTransform
import scipy.optimize
from approximation import dense_displacement_to_dct, dct_to_dense_displacement
from utils import TransformResult


def apply_transform(moving, transform):
    """
    Applies a linear or dense transformation to ``moving``.

    :param moving: The 2D-image to be transformed.
    :param transform: Any of the types :class:`skimage.transform.warp` supports for transformation,
        notably these are 3x3-matrices for projective transforms or inverse maps for dense
        transforms.
    :return: The transformation result, a 2D-image of the same dimensions as ``moving``.
    """
    return skimage.transform.warp(moving, transform)


def _get_error(moving, static):
    return np.mean(np.abs(moving - static))


def _estimate_warp_iteratively(estimate_fn, original_moving, static, n_iter, progress_bar=None):
    transform = None
    warped_moving = original_moving.copy()

    results = []

    for _ in range(n_iter):
        transform = estimate_fn(warped_moving, static, transform)
        warped_moving = apply_transform(original_moving, transform)

        # TODO there might be a setting in warp function that prevents this
        # binarize interpolated values
        warped_moving[warped_moving > 0.5] = 1
        warped_moving[warped_moving < 0.5] = 0

        # TODO think of a way to better handle energy
        # TODO HACKED!!!
        centroids, intervals = get_n_equidistant_angles_and_intervals(4)
        assignments = get_binary_assignments_from_centroids(warped_moving, centroids, intervals)
        memberships = get_memberships_from_centroids(warped_moving, centroids, intervals)
        distances = get_distance_transforms_from_binary_assignments(static)

        result = TransformResult(stacked_transform=transform,
                                 error=_get_error(warped_moving, static),
                                 energy=get_correspondences_energy(memberships, distances))
        results.append(result)

        if progress_bar:
            progress_bar.update(1)
            progress_bar.set_postfix(dict(error=result.error, energy=result.energy))

    return results


def estimate_linear_transform(moving, static, n_iter, centroids, intervals, assignments_fn,
                              reg_factor=0., progress_bar=None):
    """
    Estimates a projective transform that minimizes error of correspondences between ``moving``
    and ``static`` by transforming ``moving``. The correspondences are induced by the given
    ``centroids`` and ``intervals``, as well as the ``assignments_fn``.

    :param moving: The 2D moving image of shape (height, width).
    :param static: The 2D static image of shape (height, width).
    :param n_iter: Number of iterations.
    :param centroids: An array (n_angles,) of main directions.
    :param intervals: An array (n_angles, 2) of interval borders for each angle in ``centroids``.
    :param assignments_fn: A function that determines the assignments of ``moving`` and ``static``.
        Should be one of the functions from :mod:`distance_transform`, e.g.
        :func:`distance_transform.get_binary_assignments_from_centroids`.
    :param reg_factor: An optional factor for regularizing the least squares solution. See
        parameter ``damp`` in :func:`scipy.sparse.linalg.lsmr` for details.
    :param progress_bar: An optional progress_bar to report progress to.
    :return: A list of :class:`TransformResult` objects containing the results of each iteration.
    """
    # TODO previously we used the whole image for that to also be aware of the surroundings
    # TODO maybe use strings for assignments_fn to simplify function calls (and let module handle internal references)
    # TODO replace progress_bar by callback?
    static_assignments = get_binary_assignments_from_centroids(static, centroids, intervals)
    static_distances = get_distance_transforms_from_binary_assignments(static_assignments)
    static_directions = get_closest_feature_directions_from_binary_assignments(static_assignments)

    def estimate_fn(moving, static, previous_transform):
        moving_assignments = assignments_fn(moving, centroids, intervals)
        transform = estimate_transform_from_memberships(moving_assignments,
                                                        static_distances,
                                                        static_directions,
                                                        reg_factor,
                                                        centroids)
        # we use transform classes from skimage here, they can be concatenated with +
        return transform if previous_transform is None else transform + previous_transform

    return _estimate_warp_iteratively(estimate_fn, moving, static, n_iter, progress_bar)


def estimate_dense_displacements(moving, static, n_iter, centroids, intervals, smooth, rbf_type,
                                 assignments_fn, reduce_coeffs=None, progress_bar=None):
    """
    Estimates a dense warp field that minimizes error of correspondences between ``moving``
    and ``static`` by transforming ``moving``. The correspondences are induced by the given
    ``centroids`` and ``intervals``, as well as the ``assignments_fn``.

    :param moving: The 2D moving image of shape (height, width).
    :param static: The 2D static image of shape (height, width).
    :param n_iter: Number of iterations.
    :param centroids: An array (n_angles,) of main directions.
    :param intervals: An array (n_angles, 2) of interval borders for each angle in ``centroids``.
    :param smooth: L2-regularization factor for the RBFs.
    :param rbf_type: The type of radial basis function to use.
    :param assignments_fn: A function that determines the assignments of ``moving`` and ``static``.
        Should be one of the functions from :mod:`distance_transform`, e.g.
        :func:`distance_transform.get_binary_assignments_from_centroids`.
    :param reduce_coeffs: If not ``None``, applies a DCT to the result, truncate the coefficients
        to the given integer along both axes and transforms it back to a dense displacement.
    :param progress_bar: An optional progress_bar to report progress to.
    :return: A list of :class:`TransformResult` objects containing the results of each iteration.
    """
    static_assignments = get_binary_assignments_from_centroids(static, centroids, intervals)
    static_distances = get_distance_transforms_from_binary_assignments(static_assignments)
    static_directions = get_closest_feature_directions_from_binary_assignments(static_assignments)

    # TODO estimate reverse transform to prevent holes
    def estimate_fn(moving, static, previous_transform):
        if previous_transform is None:
            previous_transform = np.mgrid[:moving.shape[0], :moving.shape[1]]
        moving_memberships = assignments_fn(moving, centroids, intervals)
        warp_field = calculate_dense_displacements(moving_memberships, static_distances,
                                                   static_directions, smooth, rbf_type, centroids)

        if reduce_coeffs:
            dct = dense_displacement_to_dct(warp_field, reduce_coeffs)
            warp_field = dct_to_dense_displacement(dct, warp_field.shape)

        return previous_transform + warp_field

    return _estimate_warp_iteratively(estimate_fn, moving, static, n_iter, progress_bar)
