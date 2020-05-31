import numpy as np
import imageio
import argparse
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt
import skimage.transform
import scipy.interpolate
from gradient_directions import get_main_gradient_angles_and_intervals, get_gradients_in_polar_coords

parser = argparse.ArgumentParser()
parser.add_argument('feature_map_path')
args = parser.parse_args()

# load feature map and preprocess


feature_map = imageio.imread(args.feature_map_path).astype(np.float32)

if len(feature_map.shape) == 3:
    feature_map = np.mean(feature_map, axis=2)

feature_map /= feature_map.max()

centroids, intervals = get_main_gradient_angles_and_intervals(feature_map)


def get_binary_pixel_assignments(image, centroids, intervals):
    angles, magnitudes = get_gradients_in_polar_coords(image)
    # get a binary map of features for every main angle
    pixel_assignments = np.zeros((len(centroids), *image.shape), dtype=np.bool)
    mask = ~np.isclose(magnitudes, 0)

    for assignment, (low, high) in zip(pixel_assignments, intervals):
        if low < high:
            assignment[:] = mask & (low <= angles) & (angles < high)
        else:
            assignment[:] = mask & (((low - 2 * np.pi) <= angles) & (angles < high) |
                                    (low <= angles) & (angles < (high + 2 * np.pi)))

    return pixel_assignments


pixel_assignments = get_binary_pixel_assignments(feature_map, centroids, intervals)

def get_distance_transforms_from_binary_assignments(assignments):
    # calculate distance transform for the maps
    distance_transforms = np.zeros_like(assignments, dtype=np.float32)

    for distance_transform, assignment in zip(distance_transforms, assignments):
        distance_transform[:] = distance_transform_edt(1 - assignment)

    return distance_transforms


distance_transforms = get_distance_transforms_from_binary_assignments(pixel_assignments)

# take a 200x200 patch
patch_y, patch_x = 100, 400
patch_h, patch_w = 200, 200
feature_patch = feature_map[patch_y: patch_y + patch_h, patch_x: patch_x + patch_w]
feature_patch_assignments = pixel_assignments[:, patch_y: patch_y + patch_h, patch_x: patch_x + patch_w]

# and a window in the image
window_y, window_x = 500, 200
window_h, window_w = 200, 200
feature_window = feature_map[window_y: window_y + window_h, window_x: window_x + window_w]
feature_window_distances = distance_transforms[:, window_y: window_y + window_h, window_x: window_x + window_w]

plt.imshow(feature_patch - feature_window, cmap='coolwarm')
plt.show()


# determine patch energy


def get_energy(feature_masks, feature_distances):
    return sum(get_map_energy(f_mask, f_dist) for f_mask, f_dist in zip(feature_masks, feature_distances))


def get_map_energy(feature_mask, feature_distances):
    return np.sum(feature_distances[feature_mask])


for f_mask, f_dist in zip(feature_patch_assignments, feature_window_distances):
    print(get_map_energy(f_mask, f_dist))

print("=>", get_energy(feature_patch_assignments, feature_window_distances))

# determine transformation


def calculate_point_displacements(points, assignments, distances):
    displacements = np.zeros_like(points, dtype=np.float32)

    for point, displacement in zip(points, displacements):
        for angle, feature_mask, feature_distance in zip(centroids, assignments, distances):
            feature_locations = np.argwhere(feature_mask)
            feature_dist_to_corner = np.linalg.norm(feature_locations - point[None], axis=1)**(-1)
            # feature_dist_to_corner = scipy.interpolate.griddata(points=points,
            #                                                     values=np.prod(points == point, axis=1),
            #                                                     xi=feature_locations)

            feature_dt_values = feature_distance[feature_locations[:, 0], feature_locations[:, 1]]
            magnitude = np.sum(feature_dt_values * feature_dist_to_corner) / len(feature_locations)

            direction = np.array([np.cos(angle), np.sin(angle)])
            displacement += direction * magnitude

    return displacements


# apply transformation to patch


def apply_point_displacements(image, points, displacements):
    yy, xx = np.mgrid[:patch_h, :patch_w]
    grid_points = np.stack([yy, xx], axis=2)
    warp_field = -5*scipy.interpolate.griddata(points, displacements, grid_points)

    warped_image = skimage.transform.warp(image, np.array([yy + warp_field[...,0], xx + warp_field[...,1]]))
    return warped_image


corner_points = np.array([[0, 0], [0, patch_w], [patch_h, 0], [patch_h, patch_w]])
warped_feature_patch = feature_patch.copy()
warped_feature_patch_assignments = feature_patch_assignments.copy()

for _ in range(50):
    warped_feature_patch_assignments = get_binary_pixel_assignments(warped_feature_patch, centroids, intervals)
    displacements = calculate_point_displacements(points=corner_points,
                                                  assignments=warped_feature_patch_assignments,
                                                  distances=feature_window_distances)

    print(get_energy(warped_feature_patch_assignments, feature_window_distances))

    warped_feature_patch = apply_point_displacements(warped_feature_patch, corner_points, displacements)

    plt.imshow(warped_feature_patch - feature_window, cmap='coolwarm')
    plt.show()
