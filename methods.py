import numpy as np
from tqdm import tqdm
import skimage.transform
import attr
import matplotlib.pyplot as plt
from utils import plot_diff, GifExporter
from displacement import get_energy, estimate_transform_from_binary_assignments, plot_correspondences, \
    calculate_dense_displacements
from distance_transform import get_binary_assignments_from_centroids, get_distance_transforms_from_binary_assignments, \
    get_closest_feature_directions_from_binary_assignments, get_memberships_from_centroids
from gradient_directions import get_n_equidistant_angles_and_intervals
from skimage.transform import AffineTransform
import scipy.optimize


@attr.s
class TransformResult:
    stacked_transform = attr.ib()
    error = attr.ib()
    energy = attr.ib()


def apply_transform(moving, transform):
    return skimage.transform.warp(moving, transform)


def _get_error(moving, static):
    return np.mean(np.abs(moving - static))


def _estimate_warp_iteratively(estimate_fn, original_moving, static, n_iter, progress_bar=None):
    transform = None
    warped_moving = original_moving.copy()

    results = []

    #iterations = tqdm(range(n_iter), disable=not show_progressbar)
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
            progress_bar.set_postfix(dict(error=result.error))

    return results


def estimate_transform_from_correspondences(moving, static, n_iter, centroids, intervals, progress_bar=None):
    # TODO previously we used the whole image for that to also be aware of the surroundings
    static_assignments = get_binary_assignments_from_centroids(static, centroids, intervals)
    static_distances = get_distance_transforms_from_binary_assignments(static_assignments)
    static_directions = get_closest_feature_directions_from_binary_assignments(static_assignments)

    def estimate_fn(moving, static, previous_transform):
        moving_assignments = get_binary_assignments_from_centroids(moving, centroids, intervals)
        transform = estimate_transform_from_binary_assignments(moving_assignments,
                                                               static_distances,
                                                               static_directions)
        # we use transform classes from skimage here, they can be concatenated with +
        return transform if previous_transform is None else transform + previous_transform

    return _estimate_warp_iteratively(estimate_fn, moving, static, n_iter, progress_bar)


def estimate_dense_displacements_from_memberships(moving, static, n_iter, centroids, intervals, smooth, progress_bar=None):
    static_assignments = get_binary_assignments_from_centroids(static, centroids, intervals)
    static_distances = get_distance_transforms_from_binary_assignments(static_assignments)
    static_directions = get_closest_feature_directions_from_binary_assignments(static_assignments)

    def estimate_fn(moving, static, previous_transform):
        if previous_transform is None:
            previous_transform = np.mgrid[:moving.shape[0], :moving.shape[1]]
        moving_memberships = get_memberships_from_centroids(moving, centroids, intervals)
        warp_field = calculate_dense_displacements(moving_memberships, static_distances,
                                                   static_directions, smooth)
        return previous_transform + warp_field

    return _estimate_warp_iteratively(estimate_fn, moving, static, n_iter, progress_bar)


def estimate_transform_by_minimizing_energy(moving, static, n_iter, centroids, intervals, smooth, progress_bar=None):
    static_assignments = get_binary_assignments_from_centroids(static, centroids, intervals)
    static_distances = get_distance_transforms_from_binary_assignments(static_assignments)

    def opt_wrapper(transform_params):
        if np.isnan(transform_params).any():
            return np.inf

        sx, sy, rot, shear, tx, ty = transform_params
        transform = AffineTransform(scale=(1+sx,1+sy), rotation=rot, shear=shear, translation=(tx, ty))
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
        results.append(TransformResult(transform, error=_get_error(warped, static),
                                       energy=energy))

        if progress_bar:
            progress_bar.update(1)
            progress_bar.set_postfix(dict(energy=energy))

        scipy.optimize.basinhopping(opt_wrapper, x0=np.zeros(6), stepsize=1e-6, niter=n_iter, callback=callback)
    return results
