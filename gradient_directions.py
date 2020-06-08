from skimage.filters import farid_h, farid_v, gaussian
import numpy as np
from random import sample
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.cm as plt_cm

# https://arxiv.org/pdf/1601.05053.pdf
def wrapped_cauchy_kernel_density(theta, samples, weights, rho):
    """
    Evaluates the estimated density of 1D-samples at given positions ``theta``.

    :param theta: 1D-array of query positions at which the density is evaluated.
    :param samples: 1D-array of sample positions.
    :param weights: 1D-array of weights for each of the given samples.
    :param rho: Concentration parameter of the kernel. Should be from the open interval (0,1), where
        values closer to 1 produce an estimate that is more sensitive to changes.
    :return: A 1D-array with the same shape as ``theta`` with the estimated densities at the
        positions in ´`theta``.
    """
    n = len(samples)
    constant = (1 - rho ** 2) / (2 * np.pi * n)

    # [i,j] is distance of theta[i] to samples[j]
    distances = cdist(theta, samples, metric='cityblock')
    cos_distances = np.cos(distances)
    summands = weights * (1 + rho ** 2 - 2 * rho * cos_distances) ** (-1)
    densities = constant * np.sum(summands, axis=1)
    return densities / densities.sum()


def get_gradients_in_polar_coords(image):
    """
    Calculates the 2D-gradient of the given ``image`` using the 5x5 Farid kernel from
    ``skimage.filters.edges`` and converts the resulting cartesian coordinates to polar coordinates.

    :param image: a 2D-array.
    :return: A tuple ``(angles, magnitudes)`` of 2D-arrays that contain the per-pixel angles
        (between -pi and pi) and magnitudes of the gradient image.
    """
    assert len(image.shape) == 2

    dy, dx = farid_h(image), farid_v(image)
    magnitudes = np.sqrt(dy ** 2 + dx ** 2)
    angles = np.arctan2(dy, dx)
    return angles, magnitudes


def cluster_density_by_extrema(x, y):
    """
    Finds clusters in the 2D data given by 1D-arrays of x and y coordinates by splitting the data
    at the local minima and using the local maximum between two local minima as the cluster center.

    :param x: x-coordinates of the samples.
    :param y: y-coordinates of the samples.
    :return: A tuple ``(centroids, intervals)``, where ``centroids`` is a 1D-array of x-coordinates
        of the cluster centroids and ``intervals`` is a 2D-array of shape (n_centroids, 2) where
        ``intervals[i]`` is the interval of centroid ``i``.
    """
    # find peaks in density

    maxima = find_peaks(y)[0]
    minima = find_peaks(-y)[0]

    # sometimes extrema on the boundaries are not found so we add them manually
    if len(maxima) < len(minima):
        maxima = np.concatenate([[0], maxima])
    elif len(maxima) > len(minima):
        minima = np.concatenate([[0], minima])

    centroids = x[maxima]

    # if it we start with a maximum, shift the list so we start with a minimum instead
    if maxima[0] < minima[0]:
        minima = np.roll(minima, 1)

    intervals = np.stack([x[minima], x[np.roll(minima, -1)]], axis=1)

    return centroids, intervals


def get_main_gradient_angles_and_intervals(feature_map):
    """
    Calculates a clustering of the gradient angles in ``feature_map``.

    :param feature_map: A 2D-array.
    :return: A tuple ``(centroids, intervals)``, where ``centroids`` is a 1D-array of angles
        of the cluster centroids and ``intervals`` is a 2D-array of shape (n_centroids, 2) where
        ``intervals[i]`` is the angle interval of centroid ``i``.
    """
    angles, magnitudes = get_gradients_in_polar_coords(feature_map)

    # flatten
    angles = np.ravel(angles)
    magnitudes = np.ravel(magnitudes)

    # select only pixels where magnitude does not vanish
    indices = np.argwhere(~np.isclose(magnitudes, 0))[:, 0]

    # for very dense feature maps it might make sense to sample points
    # indices = sample(list(indices), k=min(len(indices), 200000))

    angles = angles[indices]
    magnitudes = magnitudes[indices]

    sample_points = np.linspace(-np.pi, np.pi, 360)

    scores = wrapped_cauchy_kernel_density(theta=sample_points[:, None],
                                           samples=angles[:, None],
                                           weights=magnitudes,
                                           rho=0.8)

    centroids, intervals = cluster_density_by_extrema(sample_points, scores)

    return centroids, intervals


# --------------
# PLOT FUNCTIONS
# --------------

def plot_polar_gradients(angles, magnitudes, ax=None):
    ax = ax or plt.gca()
    hsv = plt_cm.get_cmap('hsv')

    vis = np.zeros((*angles.shape, 3))
    vis[:] = hsv((angles + np.pi)/2/np.pi)[..., :3]
    vis *= (magnitudes[..., None] / magnitudes.max())

    ax.imshow(vis)


def plot_binary_assignments(assignments, centroids, ax=None):
    ax = ax or plt.gca()
    all_assignments = np.sum(assignments * np.arange(1, len(assignments)+1)[..., None, None], axis=0)
    angle_array = np.take(centroids, all_assignments-1)

    plot_polar_gradients(angle_array, all_assignments != 0, ax)


def plot_distance_transforms(distance_transforms, axes):
    for dt, ax in zip(distance_transforms, axes):
        ax.imshow(dt)


def plot_feature_directions(feature_directions, axes):
    for fd, ax in zip(feature_directions, axes):
        plot_polar_gradients(fd, np.ones_like(fd), ax)