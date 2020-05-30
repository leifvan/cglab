from skimage.filters import farid_h, farid_v, gaussian
import numpy as np
from random import sample
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist


# https://arxiv.org/pdf/1601.05053.pdf
def wrapped_cauchy_kernel_density(theta, locations, weights, rho):
    n = len(locations)
    constant = (1 - rho ** 2) / (2 * np.pi * n)

    # [i,j] is distance of theta[i] to locations[j]
    distances = cdist(theta, locations, metric='cityblock')
    cos_distances = np.cos(distances)
    summands = weights * (1 + rho ** 2 - 2 * rho * cos_distances) ** (-1)
    densities = constant * np.sum(summands, axis=1)
    return densities / densities.sum()


def get_gradients_in_polar_coords(image):
    assert len(image.shape) == 2

    dy, dx = farid_h(image), farid_v(image)
    magnitudes = np.sqrt(dy ** 2 + dx ** 2)
    angles = np.arctan2(dy, dx)
    return angles, magnitudes


def cluster_density_by_extrema(sample_points, scores):
    # find peaks in density

    maxima = find_peaks(scores)[0]
    minima = find_peaks(-scores)[0]

    # sometimes extrema on the boundaries are not found so we add them manually
    if len(maxima) < len(minima):
        maxima = np.concatenate([[0], maxima])
    elif len(maxima) > len(minima):
        minima = np.concatenate([[0], minima])

    centroids = sample_points[maxima]

    # if it we start with a maximum, shift the list so we start with a minimum instead
    if maxima[0] < minima[0]:
        minima = np.roll(minima, 1)

    intervals = np.stack([sample_points[minima], sample_points[np.roll(minima, -1)]], axis=1)

    return centroids, intervals


def get_main_gradient_angles_and_intervals(feature_map):
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
                                           locations=angles[:, None],
                                           weights=magnitudes,
                                           rho=0.8)

    main_gradient_angles, main_gradient_intervals = cluster_density_by_extrema(sample_points, scores)

    return main_gradient_angles, main_gradient_intervals
