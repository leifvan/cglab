from skimage.filters import sobel_h, sobel_v
import matplotlib.pyplot as plt
import matplotlib.cm as plt_cm
import imageio
import numpy as np
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import KernelDensity
from random import sample
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('feature_map_path')
args = parser.parse_args()

feature_map = imageio.imread(args.feature_map_path)

if len(feature_map.shape) == 3:
    feature_map = np.mean(feature_map, axis=2)

print(feature_map.shape)

# dy, dx = np.gradient(feature_map)
dy, dx = sobel_h(feature_map), sobel_v(feature_map)
magnitudes = (dy ** 2 + dx ** 2) ** 0.5

# norm_dy, norm_dx = dy.copy(), dx.copy()
# norm_dy[magnitudes > 0] *= magnitudes[magnitudes > 0]
# norm_dx[magnitudes > 0] *= magnitudes[magnitudes > 0]
# norm_dy[magnitudes == 0] = 0
# norm_dx[magnitudes == 0] = 0

angles = np.arctan2(dy, dx)
# plt.imshow(angles, cmap='hsv')
# plt.show()
# plt.imshow(magnitudes)
# plt.show()


hist, _ = np.histogram(a=angles, bins=360, weights=magnitudes,
                       density=True)
hist /= hist.sum()
plot_x = np.linspace(-np.pi, np.pi, len(hist))

# --------------------------------------------

data_x = np.ravel(angles)
data_w = np.ravel(magnitudes)
indices = np.argwhere(data_w > 0)[:, 0]
indices = sample(list(indices), k=min(len(indices), 200000))
# data_x = data_x[:, None]
data_x = data_x[indices][:, None]
data_w = data_w[indices]


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


y = wrapped_cauchy_kernel_density(plot_x[:, None], data_x, data_w, 0.8)

# find peaks in density
# maxima = argrelextrema(y, np.greater, mode='wrap')[0]
# minima = argrelextrema(y, np.less, mode='wrap')[0]
maxima = find_peaks(y)[0]
minima = find_peaks(-y)[0]

if len(maxima) < len(minima):
    maxima = np.concatenate([[0], maxima])
elif len(maxima) > len(minima):
    minima = np.concatenate([[0], minima])

plot_fn = plt.plot

plot_fn(plot_x, hist)
plot_fn(plot_x, y)
plot_fn(plot_x[maxima], y[maxima], 'ro')
plot_fn(plot_x[minima], y[minima], 'ko')

plt.xticks(np.arange(-4, 5) / 4 * np.pi, [f"{int(v)}°" for v in np.arange(-4, 5) / 4 * 180])
plt.yticks(plt.yticks()[0], [])

plt.show()

main_gradient_angles = plot_x[maxima]
main_gradient_intervals = np.zeros((len(maxima), 2))

# starts with a maximum
if maxima[0] < minima[0]:
    minima = np.roll(minima, 1)

main_gradient_intervals[:, 0] = plot_x[minima]
main_gradient_intervals[:, 1] = plot_x[np.roll(minima, -1)]

print(main_gradient_angles)
print(main_gradient_intervals)

# reconstruct image from main gradient directions
gradient_maps = np.zeros((len(main_gradient_angles), *feature_map.shape))

for gmap, angle in zip(gradient_maps, main_gradient_angles):
    gmap[:] = dy * np.sin(angle) + dx * np.cos(angle)
    #plt.imshow(gmap)
    #plt.show()

per_pixel_assignments = np.zeros((*feature_map.shape, 4))

labels = np.linspace(0, 1, len(main_gradient_angles), endpoint=False)
hsv_cmap = plt_cm.get_cmap('hsv')

for angle, (low, high), label in zip(main_gradient_angles, main_gradient_intervals, labels):
    if low < high:
        label_mask = (low <= angles) & (angles < high)
    else:
        label_mask = ((low - 2 * np.pi) <= angles) & (angles < high)
        label_mask = label_mask | (low <= angles) & (angles < (high + 2 * np.pi))

    per_pixel_assignments[label_mask] = hsv_cmap(label)

per_pixel_assignments *= magnitudes[..., None] / magnitudes.max()

plt.imshow(per_pixel_assignments)
plt.show()
