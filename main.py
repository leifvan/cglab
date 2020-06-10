import numpy as np
import imageio
import argparse
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt
import skimage.transform
import scipy.interpolate
from gradient_directions import get_main_gradient_angles_and_intervals, get_gradients_in_polar_coords, \
    plot_polar_gradients, plot_binary_assignments, plot_distance_transforms, plot_feature_directions

parser = argparse.ArgumentParser()
parser.add_argument('feature_map_path')
args = parser.parse_args()

# load feature map and preprocess


feature_map = imageio.imread(args.feature_map_path).astype(np.float32)

if len(feature_map.shape) == 3:
    feature_map = np.mean(feature_map, axis=2)

feature_map = skimage.transform.downscale_local_mean(feature_map, (2, 2))
feature_map /= feature_map.max()
feature_map[feature_map > 0.5] = 1
feature_map[feature_map < 0.5] = 0

angles, magnitudes = get_gradients_in_polar_coords(feature_map)
plot_polar_gradients(angles, magnitudes)
plt.show()

centroids, intervals = get_main_gradient_angles_and_intervals(feature_map)
print("Found main angles:")
for angle_degrees, interval in zip(np.degrees(centroids), np.degrees(intervals)):
    print(f"  {angle_degrees:.2f}° in [{interval[0]:.2f}°, {interval[1]:.2f}°]")


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

plot_binary_assignments(pixel_assignments, centroids)
plt.show()


def get_distance_transforms_from_binary_assignments(assignments):
    distance_transforms = np.zeros_like(assignments, dtype=np.float32)

    for distance_transform, assignment in zip(distance_transforms, assignments):
        distance_transform[:] = distance_transform_edt(1 - assignment)

    return distance_transforms


def get_closest_feature_directions_from_binary_assignments(assignments):
    directions_list = np.zeros_like(assignments, dtype=np.float32)
    grid = np.mgrid[:assignments.shape[1], :assignments.shape[2]]

    for directions, assignment in zip(directions_list, assignments):
        indices = distance_transform_edt(1 - assignment,
                                         return_distances=False,
                                         return_indices=True)
        yy, xx = grid - indices
        directions[:] = np.arctan2(yy, xx)

    return directions_list


distance_transforms = get_distance_transforms_from_binary_assignments(pixel_assignments)
feature_directions = get_closest_feature_directions_from_binary_assignments(pixel_assignments)

plot_distance_transforms(distance_transforms, plt.subplots(2, 2)[1].ravel())
plt.show()
plot_feature_directions(feature_directions, plt.subplots(2, 2)[1].ravel())
plt.show()

# # take a 200x200 patch
# patch_y, patch_x = 100, 400
# patch_h, patch_w = 200, 200
patch_y, patch_x = 30 * 2, 64 * 2
# patch_y, patch_x = 20, 10
patch_h, patch_w = 36 * 2, 36 * 2
feature_patch = feature_map[patch_y: patch_y + patch_h, patch_x: patch_x + patch_w]
feature_patch_assignments = pixel_assignments[:, patch_y: patch_y + patch_h, patch_x: patch_x + patch_w]
feature_patch_directions = feature_directions[:, patch_y: patch_y + patch_h, patch_x: patch_x + patch_w]

# # and a window in the image
# window_y, window_x = 0, 0
window_y, window_x = 2 * 2, 5 * 2
window_h, window_w = 36 * 2, 36 * 2
feature_window = feature_map[window_y: window_y + window_h, window_x: window_x + window_w]
feature_window_distances = distance_transforms[:, window_y: window_y + window_h, window_x: window_x + window_w]
feature_window_directions = feature_directions[:, window_y: window_y + window_h, window_x: window_x + window_w]

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


def calculate_point_displacements(points, angles, assignments, distances):
    displacements = np.zeros_like(points, dtype=np.float32)

    for point, displacement in zip(points, displacements):
        for angle, feature_mask, feature_distance in zip(angles, assignments, distances):
            feature_locations = np.argwhere(feature_mask)
            feature_dist_to_corner = np.linalg.norm(feature_locations / 200 - point[None] / 200, axis=1) ** (-2)
            feature_dt_values = feature_distance[feature_locations[:, 0], feature_locations[:, 1]]
            magnitude = np.sum(feature_dt_values * feature_dist_to_corner) / len(feature_locations)

            direction = np.array([np.cos(angle), -np.sin(angle)])
            displacement += direction * magnitude

    return displacements


# apply transformation to patch


def apply_point_displacements(image, points, displacements):
    yy, xx = np.mgrid[:patch_h, :patch_w]
    grid_points = np.stack([yy, xx], axis=2)
    warp_field = 0.1 * scipy.interpolate.griddata(points, displacements, grid_points)

    warped_image = skimage.transform.warp(image, np.array([yy + warp_field[..., 0], xx + warp_field[..., 1]]))
    return warped_image


def calculate_dense_displacements(angles, assignments, distances, directions, smooth):
    yy, xx = np.mgrid[:patch_h, :patch_w]

    feature_gradient_angles = np.zeros(distances.shape[1:])
    feature_gradient_magnitudes = np.zeros(distances.shape[1:])

    for angle, feature_mask, vec_magnitudes, vec_angles in zip(angles, assignments,
                                                               distances, directions):
        feature_locations = np.argwhere(feature_mask)
        feature_gradient_angles[feature_locations[:, 0],
                                feature_locations[:, 1]] = vec_angles[feature_locations[:, 0],
                                                                      feature_locations[:, 1]]
        feature_gradient_magnitudes[feature_locations[:, 0],
                                    feature_locations[:, 1]] = vec_magnitudes[feature_locations[:, 0],
                                                                              feature_locations[:, 1]]

    all_locations = np.argwhere(feature_gradient_angles)
    fy, fx = all_locations[:, 0], all_locations[:, 1]

    feature_gradient_cartesian = np.array([np.sin(feature_gradient_angles), np.cos(feature_gradient_angles)])
    feature_gradient_cartesian *= feature_gradient_magnitudes
    interpolator_y = scipy.interpolate.Rbf(fy, fx, feature_gradient_cartesian[0, fy, fx],
                                           function='linear', smooth=smooth)
    interpolator_x = scipy.interpolate.Rbf(fy, fx, feature_gradient_cartesian[1, fy, fx],
                                           function='linear', smooth=smooth)
    interpolated_y = interpolator_y(yy, xx)
    interpolated_x = interpolator_x(yy, xx)

    return np.array([interpolated_y, interpolated_x])


plot_feature_directions(feature_window_directions, plt.subplots(2, 2)[1].ravel())
plt.show()


def plot_diff(warped, target, i):
    _, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(warped, cmap='coolwarm', vmin=-1, vmax=1)
    axs[1].imshow(warped - target, cmap='coolwarm')
    axs[2].imshow(-target, cmap='coolwarm', vmin=-1, vmax=1)

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"data/plot/plot{i}.png")
    plt.show()


warped_feature_patch = feature_patch.copy()
displacement = np.mgrid[:patch_h, :patch_w].astype(np.float64)

plot_diff(warped_feature_patch, feature_window, 0)

for i in range(20):
    feature_patch_assignments = get_binary_pixel_assignments(warped_feature_patch, centroids, intervals)
    warp_field = calculate_dense_displacements(centroids, feature_patch_assignments, feature_window_distances,
                                               feature_window_directions, smooth=1e4)

    displacement += warp_field
    warped_feature_patch = skimage.transform.warp(feature_patch, displacement)
    warped_feature_patch[warped_feature_patch > 0.5] = 1
    warped_feature_patch[warped_feature_patch < 0.5] = 0

    plot_diff(warped_feature_patch, feature_window, i + 1)

with imageio.get_writer('data/plot/plot.gif', mode='I', duration=0.5) as writer:
    for i in range(20 + 1):
        filename = f"data/plot/plot{i}.png"
        image = imageio.imread(filename)
        writer.append_data(image)
