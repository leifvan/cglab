import numpy as np
import imageio
import argparse
import matplotlib.pyplot as plt
import skimage.transform
from tqdm import tqdm

from displacement import calculate_dense_displacements
from distance_transform import get_binary_assignments_from_centroids, get_distance_transforms_from_binary_assignments, \
    get_closest_feature_directions_from_binary_assignments, get_binary_assignments_from_gabor
from gradient_directions import get_main_gradient_angles_and_intervals, get_gradients_in_polar_coords, \
    plot_polar_gradients, plot_binary_assignments, plot_distance_transforms, plot_feature_directions, \
    get_n_equidistant_angles_and_intervals, apply_gabor_filters
from utils import plot_diff, GifExporter, pad_slices
from patches import find_promising_patch_pairs

parser = argparse.ArgumentParser()
parser.add_argument('feature_map_path')
args = parser.parse_args()

# -------------------------------
# load feature map and preprocess
# -------------------------------

feature_map = imageio.imread(args.feature_map_path).astype(np.float32)

if len(feature_map.shape) == 3:
    feature_map = np.mean(feature_map, axis=2)

feature_map = skimage.transform.downscale_local_mean(feature_map, (4, 4))
feature_map /= feature_map.max()
feature_map[feature_map > 0.5] = 1
feature_map[feature_map < 0.5] = 0

# ------------------------
# pick two fitting patches
# ------------------------

patch_size = 80
padding_size = 10
pairs = find_promising_patch_pairs(feature_map, patch_size=patch_size, stride=8, num_pairs=100)
# for slice_a, slice_b, _ in pairs:
#     patch_a = feature_map[slice_a]
#     patch_b = feature_map[slice_b]
#     plt.imshow(patch_a-patch_b, cmap='coolwarm')
#     plt.show()

patch_slice, window_slice, _ = pairs[0]

feature_patch = np.pad(feature_map[patch_slice], padding_size)
window_slice = pad_slices(window_slice, padding=padding_size, assert_shape=feature_map.shape)
feature_window = feature_map[window_slice]

assert feature_patch.shape == feature_window.shape


# ---------------
# get assignments
# ---------------

def get_assignments_gabor(feature_map):
    gabor_responses, angles = apply_gabor_filters(feature_map, 4, frequency=0.5, bandwidth=1.3)
    assignments = get_binary_assignments_from_gabor(gabor_responses, threshold=0.1)
    return assignments, angles


def get_assignments_clustered(feature_map):
    centroids, intervals = get_main_gradient_angles_and_intervals(feature_map)
    assignments = get_binary_assignments_from_centroids(feature_map, centroids, intervals)
    return assignments, centroids


def get_assignments_equidistant(feature_map):
    centroids, intervals = get_n_equidistant_angles_and_intervals(8)
    assignments = get_binary_assignments_from_centroids(feature_map, centroids, intervals)
    return assignments, centroids


get_assignments = get_assignments_equidistant

# angles, magnitudes = get_gradients_in_polar_coords(feature_map)
# plot_polar_gradients(angles, magnitudes)
# plt.show()

assignments, angles = get_assignments(feature_map)
plot_binary_assignments(assignments, angles)
plt.show()

# -----------------------
# get distance transforms
# -----------------------

distance_transforms = get_distance_transforms_from_binary_assignments(assignments)
feature_directions = get_closest_feature_directions_from_binary_assignments(assignments)

plot_distance_transforms(distance_transforms, plt.subplots(2, 2)[1].ravel())
plt.show()
plot_feature_directions(feature_directions, plt.subplots(2, 2)[1].ravel())
plt.show()

plt.imshow(feature_patch - feature_window, cmap='coolwarm')
plt.show()

# ------------------------------------------
# get additional information for the patches
# ------------------------------------------

feature_patch_assignments = assignments[:, patch_slice[0], patch_slice[1]]
feature_patch_directions = feature_directions[:, patch_slice[0], patch_slice[1]]
feature_window_distances = distance_transforms[:, window_slice[0], window_slice[1]]
feature_window_directions = feature_directions[:, window_slice[0], window_slice[1]]

# ---------------------
# transform iteratively
# ---------------------

plot_feature_directions(feature_window_directions, plt.subplots(2, 2)[1].ravel())
plt.show()

warped_feature_patch = feature_patch.copy()
displacement = np.mgrid[:feature_patch.shape[0], :feature_patch.shape[1]].astype(np.float64)

plot_diff(warped_feature_patch, feature_window)

gif_exporter = GifExporter()
n_iter = 20

for i in tqdm(range(n_iter)):
    feature_patch_assignments, _ = get_assignments(warped_feature_patch)
    warp_field = calculate_dense_displacements(feature_patch_assignments, feature_window_distances,
                                               feature_window_directions, smooth=1e4)

    displacement += warp_field
    warped_feature_patch = skimage.transform.warp(feature_patch, displacement, mode='constant')
    warped_feature_patch[warped_feature_patch > 0.5] = 1
    warped_feature_patch[warped_feature_patch < 0.5] = 0
    plot_diff(warped_feature_patch, feature_window)
    gif_exporter.add_current_fig()
    if i < n_iter - 1:
        plt.close()
    else:
        plt.show()

gif_exporter.save_gif("data/plot/plot.gif", duration=0.3)
