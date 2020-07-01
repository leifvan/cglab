import streamlit as st
import numpy as np
import skimage.transform
import imageio
import matplotlib.pyplot as plt
import pickle

from gradient_directions import get_n_equidistant_angles_and_intervals, get_main_gradient_angles_and_intervals, \
    plot_gradients_as_arrows
from distance_transform import get_binary_assignments_from_centroids, get_distance_transforms_from_binary_assignments, \
    get_closest_feature_directions_from_binary_assignments, get_memberships_from_centroids
from patches import find_promising_patch_pairs
from utils import plot_diff, pad_slices
from methods import estimate_transform_from_correspondences, apply_transform, \
    estimate_dense_displacements_from_memberships
from displacement import plot_correspondences


@st.cache
def feature_map():
    feature_map = imageio.imread("data/cobblestone_floor_03_AO_1k_modified.png").astype(np.float32)

    if len(feature_map.shape) == 3:
        feature_map = np.mean(feature_map, axis=2)

    feature_map = skimage.transform.downscale_local_mean(feature_map, (4, 4))
    feature_map /= feature_map.max()
    feature_map[feature_map > 0.5] = 1
    feature_map[feature_map < 0.5] = 0
    return feature_map


@st.cache
def patch_pairs():
    patch_size = 80
    return find_promising_patch_pairs(feature_map(), patch_size=patch_size, stride=16, num_pairs=1000)


@st.cache
def get_moving_and_static(number):
    patch_slice, window_slice, _ = patch_pairs()[1000 - number]
    window_slice = pad_slices(window_slice, padding=padding_size, assert_shape=feature_map().shape)
    feature_patch = np.pad(feature_map()[patch_slice], padding_size)
    feature_window = feature_map()[window_slice]
    return feature_patch, feature_window


@st.cache
def get_centroids_intervals(centroid_method, patch_position):
    moving, static = get_moving_and_static(patch_position)

    if centroid_method == 'equidistant':
        return get_n_equidistant_angles_and_intervals(num_centroids)
    elif centroid_method == 'histogram clustering':
        return get_main_gradient_angles_and_intervals(moving)


@st.cache
def run_calculation(centroid_method, transform_type, patch_position, smoothness):
    moving, static = get_moving_and_static(patch_position)
    centroids, intervals = get_centroids_intervals(centroid_method, patch_position)

    if transform_type == 'linear transform':
        results = estimate_transform_from_correspondences(moving, static, num_iterations,
                                                          centroids, intervals, False)
    elif transform_type == 'dense displacement':
        results = estimate_dense_displacements_from_memberships(moving, static, num_iterations,
                                                                centroids, intervals, smoothness, False)

    with open("results.pickle", 'wb') as pickle_file:
        pickle.dump(results, pickle_file)


'''
---
# New run
'''

"### feature map"
plt.imshow(feature_map(), cmap='bone')
st.pyplot()

padding_size = 10

patch_position = st.number_input(label='patch number', min_value=1, max_value=1000, value=250)

feature_patch, feature_window = get_moving_and_static(patch_position)
plot_diff(feature_patch, feature_window)
st.pyplot()

centroid_method = st.radio(label="centroids method", options=("equidistant",
                                                              "histogram clustering"))
if centroid_method == "equidistant":
    num_centroids = st.number_input(label="number of centroids", min_value=4, max_value=32, value=8)
elif centroid_method == 'histogram clustering':
    ... # TODO add selector for smoothness

transform_type = st.radio(label='transform type', options=("linear transform",
                                                           "dense displacement"))

if transform_type == 'dense displacement':
    smoothness = st.slider('warp field smoothness', min_value=0, max_value=10000, value=2000)
else:
    smoothness = None

num_iterations = st.number_input('number of iterations', min_value=1, max_value=100, value=20)

if st.button("run"):
    run_calculation(centroid_method, transform_type, patch_position, smoothness)

"# Results"
try:
    with open("results.pickle", 'rb') as pickle_file:
        results = pickle.load(pickle_file)

    "## unwarped"
    plot_diff(feature_patch, feature_window)
    st.pyplot()
    "## final transform"
    warped = apply_transform(feature_patch, results[-1].stacked_transform)
    plot_diff(warped, feature_window)
    st.pyplot()
    "## pick iteration"
    result_index = st.slider(label='index', min_value=0, max_value=num_iterations, step=1, value=0)
    if result_index == 0:
        plot_diff(feature_patch, feature_window)
    else:
        warped = apply_transform(feature_patch, results[result_index-1].stacked_transform)
        plot_diff(warped, feature_window)
    st.pyplot()

    _, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot([r.error for r in results])
    axs[0].set_title("error")
    axs[1].plot([r.energy for r in results])
    axs[1].set_title("energy")
    st.pyplot()

    if transform_type == 'linear transform':
        centroids, intervals = get_centroids_intervals(centroid_method, patch_position)
        static_assignments = get_binary_assignments_from_centroids(feature_window, centroids, intervals)
        plt.figure(figsize=(9,9))
        plot_correspondences(moving=warped, static=feature_window, centroids=centroids,
                             assignments=get_binary_assignments_from_centroids(warped, centroids, intervals),
                             distances=get_distance_transforms_from_binary_assignments(static_assignments),
                             directions=get_closest_feature_directions_from_binary_assignments(static_assignments))
        st.pyplot()
    elif transform_type == 'dense displacement':
        if result_index > 0:
            grid = np.mgrid[:feature_patch.shape[0], :feature_patch.shape[1]]
            dy, dx = results[result_index-1].stacked_transform - grid
            plt.figure(figsize=(9,9))
            plot_gradients_as_arrows(dy, dx, subsample=2)
            st.pyplot()

except FileNotFoundError:
    "No results found."
