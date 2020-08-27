import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, convolve
from scipy.spatial.distance import cdist
from skimage.filters import farid_h, farid_v

from utils import angle_to_rgb


def wrapped_cauchy_kernel_density(theta, samples, weights, rho):
    """
    Evaluates the estimated density of 1D-samples at given positions ``theta``.
    https://arxiv.org/pdf/1601.05053.pdf

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


def get_main_gradient_angles_and_intervals(feature_map, rho=0.8):
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
                                           rho=rho)

    centroids, intervals = cluster_density_by_extrema(sample_points, scores)

    return centroids, intervals


def get_n_equidistant_angles_and_intervals(n_angles):
    if n_angles == 1:
        return np.array([np.pi]), np.array([[np.pi, np.pi]])

    centroids = np.linspace(0, 2 * np.pi, endpoint=False, num=n_angles)
    bounds_left = (centroids - centroids[1] / 2) % (2 * np.pi)
    bounds_right = centroids + centroids[1] / 2
    intervals = np.stack([bounds_left, bounds_right], axis=1)
    # TODO check for inconsistencies with ranges [-pi, pi] and [0, 2pi], ASSERTIONS!
    return centroids - np.pi, intervals - np.pi


def get_gabor_filter(angle, sigma):
    size = max(5, int(5 * sigma - 1))  # filter size
    if size % 2 == 0:
        size += 1
    lamb = 2 * size  # wavelength
    yy, xx = np.mgrid[-size:size + 1, -size:size + 1]
    xxp = xx * np.cos(angle) + yy * np.sin(angle)
    yyp = -xx * np.sin(angle) + yy * np.cos(angle)
    gaussian = np.exp(-(xxp ** 2 + yyp ** 2) / (2 * (sigma ** 2)))
    wave = np.sin(2 * np.pi * xxp / lamb)
    filter = gaussian * wave
    filter = 2 * ((filter - filter.min()) / filter.ptp()) - 1
    return filter


def apply_gabor_filters(image, centroids, sigma):
    responses = np.zeros((len(centroids), *image.shape))
    for response, centroid in zip(responses, centroids):
        response[:] = convolve(image, get_gabor_filter(centroid, sigma), mode='same')
    return responses


# --------------
# PLOT FUNCTIONS
# --------------

def plot_gabor_filter(angle, sigma, ax=None):
    filter = -get_gabor_filter(angle, sigma)
    ax = ax or plt.gca()
    ax.imshow(filter, cmap='bone')


def plot_polar_gradients(angles, magnitudes, ax=None):
    ax = ax or plt.gca()
    vis = np.zeros((*angles.shape, 3))
    vis[:] = angle_to_rgb(angles)
    vis *= (magnitudes[..., None] / magnitudes.max())

    ax.imshow(vis)


def plot_gradients_as_arrows(dy, dx, subsample=1, scale=1, ax=None):
    original_shape = dy.shape
    dy, dx = dy[::subsample, ::subsample], dx[::subsample, ::subsample]
    ax = ax or plt.gca()
    angles = np.arctan2(dy, dx)
    yy, xx = np.mgrid[:dy.shape[1], :dy.shape[0]]
    colors = angle_to_rgb(angles).reshape(-1, 3)

    ax.quiver(xx * subsample, yy * subsample, -dx, dy, color=colors, scale=scale, scale_units='xy')
    ax.set_aspect(dy.shape[1] / dy.shape[0])
    ax.set_xlim([0, original_shape[1]-1])
    ax.set_ylim([0, original_shape[0]-1])
    ax.invert_yaxis()


def plot_binary_assignments(assignments, centroids, ax=None):
    ax = ax or plt.gca()
    all_assignments = np.sum(assignments * np.arange(1, len(assignments) + 1)[..., None, None], axis=0)
    angle_array = np.take(centroids, all_assignments - 1)

    plot_polar_gradients(angle_array, all_assignments != 0, ax)


def plot_distance_transforms(distance_transforms, axes, angles=None):
    if angles is None:
        angles = [None] * len(distance_transforms)

    for dt, ax, angle in zip(distance_transforms, axes, angles):
        ax.imshow(dt)
        if angle is not None:
            ax.set_title(f"{angle / np.pi * 180}°")


def plot_feature_directions(feature_directions, axes, angles=None):
    if angles is None:
        angles = [None] * len(feature_directions)

    for fd, ax, angle in zip(feature_directions, axes, angles):
        plot_polar_gradients(fd, np.ones_like(fd), ax)
        if angle is not None:
            ax.set_title(f"{angle / np.pi * 180}°")
