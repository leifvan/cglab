import numpy as np
import skimage.transform
import attr
from functools import partial
from displacement import get_energy, estimate_transform_from_memberships, calculate_dense_displacements
from distance_transform import get_binary_assignments_from_centroids, get_distance_transforms_from_binary_assignments, \
    get_closest_feature_directions_from_binary_assignments, get_memberships_from_centroids
from gradient_directions import get_n_equidistant_angles_and_intervals
from skimage.transform import AffineTransform
import scipy.optimize
from approximation import dense_displacement_to_dct, dct_to_dense_displacement


@attr.s
class TransformResult:
    stacked_transform = attr.ib()
    error = attr.ib()
    energy = attr.ib()


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
                                 energy=get_energy(memberships, distances))
        results.append(result)

        if progress_bar:
            progress_bar.update(1)
            progress_bar.set_postfix(dict(error=result.error, energy=result.energy))

    return results


def estimate_linear_transform(moving, static, n_iter, centroids, intervals, assignments_fn,
                              progress_bar=None):
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
                                                        static_directions)
        # we use transform classes from skimage here, they can be concatenated with +
        return transform if previous_transform is None else transform + previous_transform

    return _estimate_warp_iteratively(estimate_fn, moving, static, n_iter, progress_bar)


# TODO replace these as described in the TODO above
estimate_transform_from_binary_correspondences = partial(estimate_linear_transform,
                                                         assignments_fn=get_binary_assignments_from_centroids)
estimate_transform_from_soft_correspondences = partial(estimate_linear_transform,
                                                       assignments_fn=get_memberships_from_centroids)


def estimate_dense_displacements(moving, static, n_iter, centroids, intervals, smooth, assignments_fn,
                                 reduce_coeffs=None, progress_bar=None):
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
                                                   static_directions, smooth)

        if reduce_coeffs:
            dct = dense_displacement_to_dct(warp_field, reduce_coeffs)
            warp_field = dct_to_dense_displacement(dct, warp_field.shape)

        return previous_transform + warp_field

    return _estimate_warp_iteratively(estimate_fn, moving, static, n_iter, progress_bar)


# TODO replace as discussed above
estimate_dense_displacements_from_binary_assignments = partial(estimate_dense_displacements,
                                                               assignments_fn=get_binary_assignments_from_centroids)
estimate_dense_displacements_from_memberships = partial(estimate_dense_displacements,
                                                        assignments_fn=get_memberships_from_centroids)


def estimate_transform_by_minimizing_energy(moving, static, n_iter, centroids, intervals, smooth, progress_bar=None):
    static_assignments = get_binary_assignments_from_centroids(static, centroids, intervals)
    static_distances = get_distance_transforms_from_binary_assignments(static_assignments)

    def opt_wrapper(transform_params):
        if np.isnan(transform_params).any():
            return np.inf

        sx, sy, rot, shear, tx, ty = transform_params
        transform = AffineTransform(scale=(1 + sx, 1 + sy), rotation=rot, shear=shear, translation=(tx, ty))
        warped = apply_transform(moving, transform)

        if np.isnan(warped).any():
            return np.inf

        memberships = get_memberships_from_centroids(warped, centroids, intervals)
        return get_energy(memberships, static_distances)

    results = []

    def callback(x, energy, success):
        sx, sy, rot, shear, tx, ty = x
        transform = AffineTransform(scale=(1 + sx, 1 + sy), rotation=rot, shear=shear, translation=(tx, ty))
        warped = apply_transform(moving, transform)
        error = _get_error(warped, static)
        results.append(TransformResult(transform, error=error, energy=energy))

        if progress_bar:
            progress_bar.update(1)
            progress_bar.set_postfix(dict(energy=energy, error=error))

        scipy.optimize.basinhopping(opt_wrapper, x0=np.zeros(6), stepsize=1e-6, niter=n_iter, callback=callback)

    return results
