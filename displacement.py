import numpy as np
import scipy.interpolate


def calculate_dense_displacements(assignments, distances, directions, smooth):
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
    interpolator_y = scipy.interpolate.Rbf(fy, fx, feature_gradient_cartesian[0, fy, fx],
                                           function='linear', smooth=smooth)
    interpolator_x = scipy.interpolate.Rbf(fy, fx, feature_gradient_cartesian[1, fy, fx],
                                           function='linear', smooth=smooth)
    interpolated_y = interpolator_y(yy, xx)
    interpolated_x = interpolator_x(yy, xx)

    return np.array([interpolated_y, interpolated_x])
