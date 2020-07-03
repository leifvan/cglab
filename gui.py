import streamlit as st
import numpy as np
import skimage.transform
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors
import pickle
import random

from gradient_directions import get_n_equidistant_angles_and_intervals, get_main_gradient_angles_and_intervals, \
    plot_gradients_as_arrows, wrapped_cauchy_kernel_density, get_gradients_in_polar_coords, plot_binary_assignments
from distance_transform import get_binary_assignments_from_centroids, get_distance_transforms_from_binary_assignments, \
    get_closest_feature_directions_from_binary_assignments, get_memberships_from_centroids
from patches import find_promising_patch_pairs
from utils import plot_diff, pad_slices, get_colored_difference_image, get_slice_intersection, angle_to_rgb
from methods import estimate_transform_from_correspondences, apply_transform, \
    estimate_dense_displacements_from_memberships
from displacement import plot_correspondences


def figure_to_image():
    canvas = plt.gcf().canvas
    canvas.draw()
    buf = canvas.buffer_rgba()
    return np.asarray(buf)


@st.cache
def get_feature_map():
    feature_map = imageio.imread("data/cobblestone_floor_03_AO_1k_modified.png").astype(np.float32)

    if len(feature_map.shape) == 3:
        feature_map = np.mean(feature_map, axis=2)

    feature_map = skimage.transform.downscale_local_mean(feature_map, (4, 4))
    feature_map /= feature_map.max()
    feature_map[feature_map > 0.5] = 1
    feature_map[feature_map < 0.5] = 0
    return feature_map

feature_map = get_feature_map()


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
# Contour-based registration
'''

'''
### feature map
''' # indianred steelblue gold
st.markdown('Here we select two patches that approximately match. The moving patch is colored in '
            '<span style="color:indianred">**red**</span>, the static patch is '
            '<span style="color:steelblue">**blue**</span>, and their intersection is '
            '<span style="color:gold">**yellow**</span> (if applicable).',
            unsafe_allow_html=True)

feature_map_plot_placeholder = st.empty()

padding_size = 10

'''
### patch selection
A simple patch matching algorithm is run on the image to find two similar patches. Similarity is
measured as the MAE between the images. The number below determines which of the 1000 best pairs to
pick. The similarity decreases with higher values.
'''

patch_position = st.number_input(label='patch number', min_value=1, max_value=1000, value=250)

@st.cache
def get_patch_pairs():
    print("PATCH PAIRS!")
    patch_size = 80
    return find_promising_patch_pairs(feature_map, patch_size=patch_size, stride=16, num_pairs=1000)

patch_pairs = get_patch_pairs()
patch_slice, window_slice, _ = patch_pairs[1000 - patch_position]

@st.cache
def get_moving_and_static():
    padded_window_slice = pad_slices(window_slice, padding=padding_size, assert_shape=feature_map.shape)
    feature_patch = np.pad(feature_map[patch_slice], padding_size)
    feature_window = feature_map[padded_window_slice]
    return feature_patch, feature_window

moving, static = get_moving_and_static()
intersection_slice = get_slice_intersection(patch_slice, window_slice)

@st.cache
def plot_colored_feature_map():
    colored_map = 1-np.tile(feature_map[...,None], reps=(1,1,3)) * 0.3
    colored_map[patch_slice[0], patch_slice[1], :] = get_colored_difference_image(moving=feature_map[patch_slice])
    colored_map[window_slice[0], window_slice[1], :] = get_colored_difference_image(static=feature_map[window_slice])
    colored_map[intersection_slice[0], intersection_slice[1], :] = get_colored_difference_image(static=feature_map[intersection_slice],
                                                                                                moving=feature_map[intersection_slice])
    plt.imshow(colored_map)
    return figure_to_image()

feature_map_plot_placeholder.image(image=plot_colored_feature_map())

'''
This is an overlay of the two chosen patches.
'''

@st.cache
def plot_moving_static_diff():
    plot_diff(moving, static)
    return figure_to_image()

st.image(image=plot_moving_static_diff(), use_column_width=True)


'''
### Finding dominant gradient directions
Now we are looking for the main directions of gradients in the image. For each main direction we also
need an interval s.t. every angle in that interval is assigned to the main direction.
'''
centroid_method = st.radio(label="centroids method", options=("equidistant",
                                                              "histogram clustering"))
num_centroids = kde_rho = None
if centroid_method == "equidistant":
    '''
    Here we simply choose $n$ equidistant directions and intervals.
    '''
    num_centroids = st.number_input(label="number of angles", min_value=2, max_value=32, value=8)
elif centroid_method == 'histogram clustering':
    r'''
    Use a direct kernel density estimation on the angles and take the maxima of the resulting
    function as the dominant gradient directions. As the angle histogram is periodic, we use a
    wrapped Cauchy kernel with density function
    $$
    f(\theta ; \mu, \rho ) = \frac{1}{2\pi} \frac{1-\rho^2}{1+\rho^2-2\rho\cos(\theta-\mu)}
    $$
    which leads to the kernel
    $$
    \hat{f}(\theta; \rho) = \frac{1}{N} \sum_{j=1}^N f(\theta; \theta_j, \rho).
    $$
    $\rho\in (0,1)$ is the smoothness parameter, where higher values lead to a less-smoothed estimate.
    '''
    kde_rho = st.slider(label='rho', min_value=0., max_value=1., value=0.8)


@st.cache
def get_centroids_intervals():
    if centroid_method == 'equidistant':
        return get_n_equidistant_angles_and_intervals(num_centroids)
    elif centroid_method == 'histogram clustering':
        return get_main_gradient_angles_and_intervals(moving, kde_rho)


centroids, intervals = get_centroids_intervals()
centroids_degrees = [f"{-(c/np.pi*180+180)%360:.0f}°" for c in centroids]
centroids_colors = angle_to_rgb(centroids)

def write_centroid_legend():
    centroid_colors = [matplotlib.colors.to_hex(c) for c in centroids_colors]
    items = [f'<span style="color:{c}">{cd}</span>' for c, cd in zip(centroid_colors, centroids_degrees)]
    html = '<div style="background-color: black; color:white; float:left">Angles: '+', '.join(items)+'</div>'
    st.markdown(html, unsafe_allow_html=True)

@st.cache
def get_kde_plot_data():
    # TODO this is just copy-and-pasted from the code in gradient_directions.py
    angles, magnitudes = get_gradients_in_polar_coords(moving)

    # flatten
    angles = np.ravel(angles)
    magnitudes = np.ravel(magnitudes)

    # select only pixels where magnitude does not vanish
    indices = np.argwhere(~np.isclose(magnitudes, 0))[:, 0]

    # for very dense feature maps it might make sense to sample points
    indices = random.sample(list(indices), k=min(len(indices), 2000))

    angles = angles[indices]
    magnitudes = magnitudes[indices]

    sample_points = np.linspace(-np.pi, np.pi, 360)
    scores = wrapped_cauchy_kernel_density(theta=sample_points[:, None],
                                           samples=angles[:, None],
                                           weights=magnitudes,
                                           rho=kde_rho)
    return sample_points, scores

@st.cache
def plot_angles():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')

    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks(centroids)
    ax.set_xticklabels(centroids_degrees)

    for angle, (ilow, ihigh), color in zip(centroids, intervals, centroids_colors):
        ax.plot([angle, angle], [0, 1], c=color)
        if ilow > ihigh:
            ihigh += 2*np.pi
        ax.fill_between(np.linspace(ilow, ihigh, num=3), 0, 2, color=color, alpha=0.1)

    ax.set_ylim(0,1)
    ax.set_yticks([])
    ax.set_theta_zero_location('W')
    ax.set_theta_direction(-1)

    if centroid_method == 'histogram clustering':
        sample_points, scores = get_kde_plot_data()
        ax.plot(sample_points, scores)
        ax.set_ylim(0, scores.max())

    return figure_to_image()

''
st.image(image=plot_angles())


'''
### Binary assignments
'''

write_centroid_legend()

@st.cache
def plot_assignments():
    _, axs = plt.subplots(1,2, figsize=(8,4))
    moving_assignments = get_binary_assignments_from_centroids(moving, centroids, intervals)
    static_assignments = get_binary_assignments_from_centroids(static, centroids, intervals)
    plot_binary_assignments(moving_assignments, centroids, axs[0])
    axs[0].set_title("moving")
    plot_binary_assignments(static_assignments, centroids, axs[1])
    axs[1].set_title("static")

    return figure_to_image()

st.image(image=plot_assignments(), use_column_width=True)

'''
### Memberships
AKA softassignments.
'''

picked_angle = st.selectbox(label='angle', options=centroids_degrees)
picked_angle_index = centroids_degrees.index(picked_angle)

write_centroid_legend()

@st.cache
def plot_memberships():
    plt.figure()
    memberships = get_memberships_from_centroids(moving, centroids, intervals)
    color = centroids_colors[picked_angle_index]
    membership_image = np.tile(moving[...,None], reps=(1,1,3)) * 0.2 + 0.8 * memberships[picked_angle_index,...,None] * color
    plt.imshow(membership_image)
    return figure_to_image()

st.image(image=plot_memberships())


exit(0)

# TODO plot assignments

'''
### Type of transform
Based on the directions, we can now fit a transformation.
'''


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
