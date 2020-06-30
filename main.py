import argparse

import imageio
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import skimage.transform
from tqdm import tqdm

from displacement import calculate_dense_displacements, plot_correspondences, estimate_transform_from_binary_assignments
from distance_transform import get_binary_assignments_from_centroids, get_distance_transforms_from_binary_assignments, \
    get_binary_assignments_from_gabor, get_memberships_from_centroids, \
    get_closest_feature_directions_from_distance_transforms, get_closest_feature_directions_from_binary_assignments
from gradient_directions import get_main_gradient_angles_and_intervals, plot_binary_assignments, \
    plot_distance_transforms, plot_feature_directions, \
    get_n_equidistant_angles_and_intervals, apply_gabor_filters, plot_gradients_as_arrows
from patches import find_promising_patch_pairs
from utils import plot_diff, GifExporter, pad_slices, tight_layout_with_suptitle, get_quadratic_subplot_for_n_axes

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
pairs = find_promising_patch_pairs(feature_map, patch_size=patch_size, stride=16, num_pairs=439)
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


centroids, intervals = get_n_equidistant_angles_and_intervals(4)


def get_assignments_equidistant(feature_map):
    assignments = get_binary_assignments_from_centroids(feature_map, centroids, intervals)
    return assignments, centroids


get_assignments = get_assignments_equidistant

# angles, magnitudes = get_gradients_in_polar_coords(feature_map)
# plot_polar_gradients(angles, magnitudes)
# plt.show()

assignments, angles = get_assignments(feature_map)
# plot_binary_assignments(assignments, angles)
# plt.title("binary assignments")
# plt.show()

# -----------------------
# get distance transforms
# -----------------------

distance_transforms = get_distance_transforms_from_binary_assignments(assignments)
feature_directions = get_closest_feature_directions_from_distance_transforms(distance_transforms)

# plot_distance_transforms(distance_transforms,
#                          axes=get_quadratic_subplot_for_n_axes(len(centroids), True),
#                          angles=centroids)
# plt.suptitle("distance transforms")
# tight_layout_with_suptitle()
# plt.show()
# plot_feature_directions(feature_directions,
#                         axes=get_quadratic_subplot_for_n_axes(len(centroids), True),
#                         angles=centroids)
# plt.suptitle("feature directions")
# tight_layout_with_suptitle()
# plt.show()
#
# plt.imshow(feature_patch - feature_window, cmap='coolwarm')
# plt.show()

# ------------------------------------------
# get additional information for the patches
# ------------------------------------------

# TODO its questionable which approach to choose here
# feature_window_distances = distance_transforms[:, window_slice[0], window_slice[1]]
# feature_window_directions = feature_directions[:, window_slice[0], window_slice[1]]
feature_window_assignments = assignments[:, window_slice[0], window_slice[1]]
feature_window_distances = get_distance_transforms_from_binary_assignments(feature_window_assignments)
feature_window_directions = get_closest_feature_directions_from_binary_assignments(feature_window_assignments)
# feature_window_directions = get_closest_feature_directions_from_distance_transforms(feature_window_distances)

# ---------------------
# transform iteratively
# ---------------------

# axs = get_quadratic_subplot_for_n_axes(len(centroids), True)
# plot_distance_transforms(feature_window_distances, axs, angles=centroids)
# axs[-1].imshow(feature_window)
# plt.suptitle("window distance transforms")
# tight_layout_with_suptitle()
# plt.show()
#
# plot_feature_directions(feature_window_directions,
#                         axes=get_quadratic_subplot_for_n_axes(len(centroids), True),
#                         angles=centroids)
# plt.suptitle("window feature directions")
# tight_layout_with_suptitle()
# plt.show()

warped_feature_patch = feature_patch.copy()
grid = np.mgrid[:feature_patch.shape[0], :feature_patch.shape[1]].astype(np.float64)
displacement = np.zeros_like(grid)

# TODO the initial / unwarped diff is not in the exported gif
plot_diff(warped_feature_patch, feature_window)
plt.show()

gif_exporter = GifExporter()
gif_exporter_correspondences = GifExporter()
n_iter = 40

errors = np.zeros(n_iter)

transform = None

for i in tqdm(range(n_iter)):
    smooth = max(400, 6e3 - 200 * i)
    feature_patch_memberships = get_memberships_from_centroids(warped_feature_patch, centroids, intervals)
    feature_patch_assignments = get_binary_assignments_from_centroids(warped_feature_patch, centroids, intervals)

    plt.figure(figsize=(8, 8))
    plot_correspondences(warped_feature_patch, feature_window, centroids, feature_patch_memberships,
                         feature_window_distances, feature_window_directions)
    gif_exporter_correspondences.add_current_fig()
    plt.close()

    new_transform = estimate_transform_from_binary_assignments(feature_patch_assignments,
                                                               feature_window_distances,
                                                               feature_window_directions)
    transform = new_transform if transform is None else new_transform + transform
    #transform = new_transform if transform is None else new_transform @ transform
    warped_feature_patch = skimage.transform.warp(feature_patch, transform)
    # plot_diff(patch_t_warped, feature_window)
    # plt.show()
    # exit(0)

    # warp_field = calculate_dense_displacements(feature_patch_memberships, feature_window_distances,
    #                                            feature_window_directions, smooth=smooth)
    #
    # displacement += warp_field
    # warped_feature_patch = skimage.transform.warp(feature_patch, grid + displacement, mode='constant')
    warped_feature_patch[warped_feature_patch > 0.5] = 1
    warped_feature_patch[warped_feature_patch < 0.5] = 0

    errors[i] = np.mean(np.abs(warped_feature_patch - feature_window))
    # plotting for gif output
    _, axs = plt.subplots(2, 3, figsize=(12, 9))
    plot_diff(warped_feature_patch, feature_window, axs=axs[0])
    # TODO normalize displacement plotting of all figures
    # plot_gradients_as_arrows(*displacement, subsample=4, ax=axs[1, 0])
    # plot_gradients_as_arrows(*warp_field, subsample=4, ax=axs[1, 1])
    plt.suptitle(f"{i + 1} / {n_iter}, smooth={smooth}")
    axs[1, 2].plot(errors[:i + 1])
    axs[1, 2].set_xlim(0, n_iter)
    axs[1, 2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # axs[1, 2].set_ylim(errors[:i+1].min(), errors[0])
    # axs[1, 2].set_ylim(0, errors[0])
    tight_layout_with_suptitle()
    gif_exporter.add_current_fig()
    # if i < n_iter - 1:
    #     plt.close()
    # else:
    plt.show()

    # TODO save first and list image separately

gif_exporter.save_gif("data/plot/plot.gif", duration=0.5)
gif_exporter_correspondences.save_gif("data/plot/correspondences.gif", duration=0.5)
