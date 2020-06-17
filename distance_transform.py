import numpy as np
from scipy.ndimage import distance_transform_edt

from gradient_directions import get_gradients_in_polar_coords


def get_binary_assignments_from_centroids(image, centroids, intervals):
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
    assignments = np.zeros_like(responses, dtype=np.bool)
    argmax = np.argmax(np.abs(responses), axis=0)
    for i, assignment in enumerate(assignments):
        assignment[:] = (argmax == i) & ~np.isclose(responses[i], 0, atol=threshold)
    return assignments


def get_distance_transforms_from_binary_assignments(assignments):
    distance_transforms = np.zeros_like(assignments, dtype=np.float32)

    for distance_transform, assignment in zip(distance_transforms, assignments):
        distance_transform[:] = distance_transform_edt(1 - assignment)

    return distance_transforms


def get_closest_feature_directions_from_binary_assignments(assignments):
    directions_list = np.zeros_like(assignments, dtype=np.float32)
    grid = np.mgrid[:assignments.shape[1], :assignments.shape[2]]

    for directions, assignment in zip(directions_list, assignments):
        indices = distance_transform_edt(1 - assignment,
                                         return_distances=False,
                                         return_indices=True)
        yy, xx = grid - indices
        directions[:] = np.arctan2(yy, xx)

    return directions_list