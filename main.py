import numpy as np
import imageio
import argparse
import matplotlib.pyplot as plt
import skimage.transform
from tqdm import tqdm

from displacement import calculate_dense_displacements
from distance_transform import get_binary_assignments_from_centroids, get_distance_transforms_from_binary_assignments, \
    get_closest_feature_directions_from_binary_assignments, get_binary_assignments_from_gabor, \
    get_memberships_from_centroids
from gradient_directions import get_main_gradient_angles_and_intervals, get_gradients_in_polar_coords, \
    plot_polar_gradients, plot_binary_assignments, plot_distance_transforms, plot_feature_directions, \
    get_n_equidistant_angles_and_intervals, apply_gabor_filters, plot_gradients_as_arrows
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
pairs = find_promising_patch_pairs(feature_map, patch_size=patch_size, stride=16, num_pairs=124)
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


centroids, intervals = get_n_equidistant_angles_and_intervals(8)

def get_assignments_equidistant(feature_map):
    assignments = get_binary_assignments_from_centroids(feature_map, centroids, intervals)
    return assignments, centroids


get_assignments = get_assignments_equidistant

# angles, magnitudes = get_gradients_in_polar_coords(feature_map)
# plot_polar_gradients(angles, magnitudes)
# plt.show()

assignments, angles = get_assignments(feature_map)
plot_binary_assignments(assignments, angles)
plt.title("binary assignments")
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

feature_window_distances = distance_transforms[:, window_slice[0], window_slice[1]]
feature_window_directions = feature_directions[:, window_slice[0], window_slice[1]]

# ---------------------
# transform iteratively
# ---------------------

plot_feature_directions(feature_window_directions, plt.subplots(3, 3)[1].ravel())
plt.show()

warped_feature_patch = feature_patch.copy()
grid = np.mgrid[:feature_patch.shape[0], :feature_patch.shape[1]].astype(np.float64)
displacement = np.zeros_like(grid)

# TODO the initial / unwarped diff is not in the exported gif
plot_diff(warped_feature_patch, feature_window)
plt.show()

gif_exporter = GifExporter()
n_iter = 50

for i in tqdm(range(n_iter)):
    smooth = max(1000, 5e3 - 200 * i)
    feature_patch_memberships = get_memberships_from_centroids(warped_feature_patch, centroids, intervals)

    # _, axs = plt.subplots(3,3,figsize=(9,9))
    # axs = axs.ravel()
    #
    # plot_polar_gradients(*get_gradients_in_polar_coords(warped_feature_patch), ax=axs[-1])
    # for ax, angle, membership in zip(axs, centroids, feature_patch_memberships):
    #     ax.imshow(membership, cmap='bone')
    #     #plot_polar_gradients(feature_window_directions[i], np.ones_like(feature_window_directions[i]), ax)
    #
    # plt.show()

    warp_field = calculate_dense_displacements(feature_patch_memberships, feature_window_distances,
                                               feature_window_directions, smooth=smooth)

    displacement += warp_field
    warped_feature_patch = skimage.transform.warp(feature_patch, grid + displacement, mode='constant')
    warped_feature_patch[warped_feature_patch > 0.5] = 1
    warped_feature_patch[warped_feature_patch < 0.5] = 0

    # plotting for gif output
    _, axs = plt.subplots(2, 3, figsize=(12, 8))
    plot_diff(warped_feature_patch, feature_window, axs=axs[0])
    # TODO normalize displacement plotting of all figures
    plot_gradients_as_arrows(*displacement, subsample=2, ax=axs[1, 0])
    plot_gradients_as_arrows(*warp_field, subsample=2, ax=axs[1, 1])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f"{i + 1} / {n_iter}, smooth={smooth}")
    gif_exporter.add_current_fig()
    if i < n_iter - 1:
        plt.close()
    else:
        plt.show()

    # TODO save first and list image separately

gif_exporter.save_gif("data/plot/plot.gif", duration=0.5)
