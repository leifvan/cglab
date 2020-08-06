import numpy as np
import numba as nb
from scipy.ndimage import distance_transform_edt
from gradient_directions import get_gradients_in_polar_coords, apply_gabor_filters


def assert_assignments_binary(fn):
    def _assert_assignments_binary(image, centroids, intervals, **kwargs):
        assignments = fn(image, centroids, intervals, **kwargs)
        assert np.all(assignments.sum(axis=0) <= 1)
        return assignments
    return _assert_assignments_binary

# FIXME add threshold to docstring
@assert_assignments_binary
def get_binary_assignments_from_centroids(image, centroids, intervals, threshold=1e-8):
    """
    Assigns each pixel of ``image`` with non-vanishing gradient to one of the angles in
    ``centroids`` based on the given intervals.

    :param image: The 2D input image of shape (height, width).
    :param centroids: An array (n_angles,) of angles.
    :param intervals: An array (n_angles, 2) of interval borders for each angle in ``centroids``.
    :return: An array (n_angles, height, width) of binary maps for each centroid that assign a
        pixel with non-vanishing gradient to exactly one of the centroids, i.e. if index [k,i,j] is
        1, the pixel ``image[i,j]`` is assigned to angle ``centroids[k]``.
    """
    angles, magnitudes = get_gradients_in_polar_coords(image)
    # get a binary map of features for every main angle
    pixel_assignments = np.zeros((len(centroids), *image.shape), dtype=np.bool)
    mask = ~np.isclose(magnitudes, 0, atol=threshold)

    for assignment, (low, high) in zip(pixel_assignments, intervals):
        if low < high:
            assignment[:] = mask & (low <= angles) & (angles < high)
        else:
            assignment[:] = mask & (((low - 2 * np.pi) <= angles) & (angles < high) |
                                    (low <= angles) & (angles < (high + 2 * np.pi)))

    return pixel_assignments


@assert_assignments_binary
def get_binary_assignments_from_gabor(image, centroids, intervals, sigma, threshold=1e-2):
    # correct the direction
    responses = -apply_gabor_filters(image, centroids, sigma)
    responses /= responses.max()

    argmax = np.argmax(responses, axis=0)
    assignments = np.zeros((len(centroids), *image.shape), dtype=np.bool)
    for i, assignment in enumerate(assignments):
        assignment[:] = (argmax == i) & (~np.isclose(responses[i], 0, atol=threshold))
    return assignments


def get_memberships_from_gabor(image, centroids, intervals, sigma, threshold=1e-2):
    responses = -apply_gabor_filters(image, centroids, sigma)
    responses /= responses.max()
    responses[responses <= threshold] = 0
    return responses


def get_distance_transforms_from_binary_assignments(assignments):
    """
    Calculates the euclidean distance transform for each of the binary assignments given.

    :param assignments: An array (n_angles, height, width) with binary assignments for each pixel.
    :return: An array (n_angles, height, width) of distance transforms, where index [k,i,j] contains
        the euclidean distance of pixel [i,j] to its nearest edge pixel.
    """
    distance_transforms = np.zeros_like(assignments, dtype=np.float32)

    for distance_transform, assignment in zip(distance_transforms, assignments):
        distance_transform[:] = distance_transform_edt(1 - assignment)

    return distance_transforms


def get_closest_feature_directions_from_binary_assignments(assignments):
    """
    Calculates the angle between each pixel to its closest pixel in assignments.

    :param assignments: An array (n_angles, height, width) with binary assignments for each pixel.
    :return: An array (n_angles, height, width) of angles, where index [k,i,j] contains the
        angle of pixel [i,j] to its closest edge pixel.
    """

    directions_list = np.zeros_like(assignments, dtype=np.float32)
    grid = np.mgrid[:assignments.shape[1], :assignments.shape[2]]

    for directions, assignment in zip(directions_list, assignments):
        indices = distance_transform_edt(1 - assignment,
                                         return_distances=False,
                                         return_indices=True)
        yy, xx = grid - indices
        directions[:] = np.arctan2(yy, xx)

    return directions_list


def get_closest_feature_directions_from_distance_transforms(distance_transforms):
    directions_list = np.zeros_like(distance_transforms)
    for directions, distance_transform in zip(directions_list, distance_transforms):
        directions[:] = get_gradients_in_polar_coords(distance_transform)[0]
    return directions_list


@nb.vectorize
def linear_ramp_membership(x, c_left, c, c_right):
    if x < c_left or x > c_right:
        return 0
    elif x <= c:
        return 1 - (x-c) / (c_left - c)
    else:  # x > c
        return 1 - (x-c) / (c_right - c)


def get_memberships_from_centroids(image, centroids, intervals):
    angles, magnitudes = get_gradients_in_polar_coords(image)
    memberships = np.zeros((len(centroids), *image.shape))
    mask = ~np.isclose(magnitudes, 0, atol=0.2)

    assert np.all(-np.pi <= centroids) and np.all(centroids <= np.pi)
    assert np.all(-np.pi <= angles) and np.all(angles <= np.pi)


    # TODO scale with interval widths
    for membership, prev_c, cur_c, next_c in zip(memberships, np.roll(centroids,1),
                                                 centroids, np.roll(centroids,-1)):
        if prev_c > cur_c:
            prev_c -= 2*np.pi
        elif next_c < cur_c:
            next_c += 2*np.pi

        assert prev_c < cur_c < next_c

        membership[mask] = linear_ramp_membership(angles[mask], prev_c, cur_c, next_c) ** 2
        #membership[mask] = np.maximum(0., np.cos(centroid-(angles[mask] + np.pi)))

    # TODO is this needed? should be obsolete when scaling with interval widths
    # normalize membership of each pixel to 1
    # memberships[:,mask] /= np.linalg.norm(memberships[:,mask], axis=0)

    return memberships
