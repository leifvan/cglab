import numpy as np
import scipy.interpolate


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
    :return:
    """
    height, width = distances.shape[1:3]
    yy, xx = np.mgrid[:height, :width]

    feature_gradient_angles = np.zeros(distances.shape[1:])
    feature_gradient_magnitudes = np.zeros(distances.shape[1:])

    for feature_mask, vec_magnitudes, vec_angles in zip(assignments, distances, directions):
        feature_locations = np.argwhere(feature_mask)
        feature_gradient_angles[feature_locations[:, 0],
                                feature_locations[:, 1]] = vec_angles[feature_locations[:, 0],
                                                                      feature_locations[:, 1]]
        feature_gradient_magnitudes[feature_locations[:, 0],
                                    feature_locations[:, 1]] = vec_magnitudes[feature_locations[:, 0],
                                                                              feature_locations[:, 1]]

    all_locations = np.argwhere(feature_gradient_angles)
    fy, fx = all_locations[:, 0], all_locations[:, 1]

    feature_gradient_cartesian = np.array([np.sin(feature_gradient_angles), np.cos(feature_gradient_angles)])
    feature_gradient_cartesian *= feature_gradient_magnitudes
    # TODO check if we should change the interpolation mode='N-D' (because default is '1-D')
    interpolator_y = scipy.interpolate.Rbf(fy, fx, feature_gradient_cartesian[0, fy, fx],
                                           function='linear', smooth=smooth)
    interpolator_x = scipy.interpolate.Rbf(fy, fx, feature_gradient_cartesian[1, fy, fx],
                                           function='linear', smooth=smooth)
    interpolated_y = interpolator_y(yy, xx)
    interpolated_x = interpolator_x(yy, xx)

    return np.array([interpolated_y, interpolated_x])
