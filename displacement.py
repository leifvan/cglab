import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from gradient_directions import plot_polar_gradients, plot_gradients_as_arrows


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
    :return: An array (2, height, width) of y and x displacements for each pixel, i.e. [:,i,j] is
        the displacement (y,x) of pixel [i,j].
    """
    height, width = distances.shape[1:3]
    yy, xx = np.mgrid[:height, :width]

    feature_gradient_cartesian = np.array([np.sin(directions), np.cos(directions)])
    feature_gradient_cartesian *= distances * assignments
    feature_gradient_cartesian = feature_gradient_cartesian.sum(axis=1)

    all_locations = np.argwhere(np.logical_or.reduce(assignments, axis=0))
    fy, fx = all_locations[:, 0], all_locations[:, 1]

    transposed_coords = np.transpose(feature_gradient_cartesian[:, fy, fx])
    interpolator = scipy.interpolate.Rbf(fy, fx, transposed_coords, function='linear',
                                         smooth=smooth, mode='N-D')
    interpolated = interpolator(yy, xx)
    return np.moveaxis(interpolated, 2, 0)
