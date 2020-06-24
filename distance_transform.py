import numpy as np
from scipy.ndimage import distance_transform_edt
from gradient_directions import get_gradients_in_polar_coords


def get_binary_assignments_from_centroids(image, centroids, intervals):
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
    mask = ~np.isclose(magnitudes, 0)

    for assignment, (low, high) in zip(pixel_assignments, intervals):
        if low < high:
            assignment[:] = mask & (low <= angles) & (angles < high)
        else:
            assignment[:] = mask & (((low - 2 * np.pi) <= angles) & (angles < high) |
                                    (low <= angles) & (angles < (high + 2 * np.pi)))

    return pixel_assignments


def get_binary_assignments_from_gabor(responses, threshold):
    """
    Assigns each pixel with non-vanishing response to the filter with maximum absolute response.

    :param responses: An array (n_filters, height, width) of filter responses from gabor filters.
    :param threshold: All pixels with a response below this value will not be considered.
    :return: An array (n_filters, height, width) of binary maps for each filter that assign a
        pixel with non-vanishing response to exactly one of the filters, i.e. if index [k,i,j] is
        1, the pixel [i,j] is assigned to filter k.
    """

    # TODO check if we can create gabor filters that aren't 180°-symmetric
    assignments = np.zeros_like(responses, dtype=np.bool)
    argmax = np.argmax(np.abs(responses), axis=0)
    for i, assignment in enumerate(assignments):
        assignment[:] = (argmax == i) & ~np.isclose(responses[i], 0, atol=threshold)
    return assignments


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


def get_memberships_from_centroids(image, centroids, intervals):
    angles, magnitudes = get_gradients_in_polar_coords(image)
    memberships = np.zeros((len(centroids), *image.shape))
    mask = ~np.isclose(magnitudes, 0, atol=0.2)

    # TODO scale with interval widths
    for membership, centroid in zip(memberships, centroids):
        membership[mask] = np.maximum(0., np.cos(centroid-(angles[mask] + np.pi)))

    # TODO is this needed? should be obsolete when scaling with interval widths
    # normalize membership of each pixel to 1
    #memberships[:,mask] /= np.linalg.norm(memberships[:,mask], axis=0)

    return memberships
