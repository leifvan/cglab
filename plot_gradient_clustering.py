import matplotlib.pyplot as plt
import matplotlib.cm as plt_cm
import imageio
import numpy as np
from gradient_directions import get_gradients_in_polar_coords, \
    get_main_gradient_angles_and_intervals, wrapped_cauchy_kernel_density, \
    cluster_density_by_extrema

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('feature_map_path')
args = parser.parse_args()

# load feature map and preprocess


feature_map = imageio.imread(args.feature_map_path).astype(np.float32)

if len(feature_map.shape) == 3:
    feature_map = np.mean(feature_map, axis=2)


feature_map /= feature_map.max()


# get polar gradients


angles, magnitudes = get_gradients_in_polar_coords(feature_map)
plt.imshow(magnitudes)
plt.show()


mask = ~np.isclose(magnitudes, 0)

hist, _ = np.histogram(a=angles[mask], bins=360, range=(-np.pi, np.pi),
                       weights=magnitudes[mask],
                       density=True)
hist /= hist.sum()
plot_x = np.linspace(-np.pi, np.pi, len(hist))

y = wrapped_cauchy_kernel_density(theta=plot_x[:,None],
                                  samples=np.ravel(angles[mask])[:, None],
                                  weights=magnitudes[mask],
                                  rho=0.8)

centroids, intervals = cluster_density_by_extrema(plot_x, y)
centroids_idx = ((centroids + np.pi) / (2*np.pi) * 360).astype(np.int)
intervals_idx = ((intervals + np.pi) / (2*np.pi) * 360).astype(np.int)

plot_fn = plt.plot

plot_fn(plot_x, hist)
plot_fn(plot_x, y)
plot_fn(centroids, y[centroids_idx], 'ro')
plot_fn(intervals[:,0], y[intervals_idx[:,0]], 'ko')

plt.xticks(np.arange(-4, 5) / 4 * np.pi, [f"{int(v)}°" for v in np.arange(-4, 5) / 4 * 180])
plt.yticks(plt.yticks()[0], [])

plt.show()

# get gradients and plot segmentation

main_gradient_angles, main_gradient_intervals = get_main_gradient_angles_and_intervals(feature_map)
print("Found",len(main_gradient_angles), "main gradient directions")

per_pixel_assignments = np.zeros((*feature_map.shape, 3))

labels = np.linspace(0, 1, len(main_gradient_angles), endpoint=False)
hsv_cmap = plt_cm.get_cmap('hsv')

# for each angle, find corresponding pixels and colorize them
for angle, (low, high) in zip(main_gradient_angles, main_gradient_intervals):
    if low < high:
        label_mask = (mask) & (low <= angles) & (angles < high)
    else:
        label_mask = (mask) & ((low - 2 * np.pi) <= angles) & (angles < high)
        label_mask = label_mask | (mask) & (low <= angles) & (angles < (high + 2 * np.pi))

    per_pixel_assignments[label_mask] = hsv_cmap((angle + np.pi)/2/np.pi)[:3]

#per_pixel_assignments *= magnitudes[..., None] / magnitudes.max()

plt.imshow(per_pixel_assignments)
plt.show()
