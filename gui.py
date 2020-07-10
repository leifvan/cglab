import streamlit as st
import numpy as np
import skimage.transform
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors
import pickle
import random
import attr
import os
import string
import time

from gradient_directions import get_n_equidistant_angles_and_intervals, get_main_gradient_angles_and_intervals, \
    plot_gradients_as_arrows, wrapped_cauchy_kernel_density, get_gradients_in_polar_coords, plot_binary_assignments
from distance_transform import get_binary_assignments_from_centroids, get_distance_transforms_from_binary_assignments, \
    get_closest_feature_directions_from_binary_assignments, get_memberships_from_centroids
from patches import find_promising_patch_pairs
from utils import plot_diff, pad_slices, get_colored_difference_image, get_slice_intersection, angle_to_rgb
from methods import estimate_transform_from_correspondences, apply_transform, \
    estimate_dense_displacements_from_memberships
from displacement import plot_correspondences


@attr.s
class RunConfiguration:
    # TODO also save feature map (path)
    patch_position: int = attr.ib()
    centroid_method: str = attr.ib()
    num_centroids: int = attr.ib()
    kde_rho: float = attr.ib()
    transform_type: str = attr.ib()
    smoothness: int = attr.ib()
    num_iterations: int = attr.ib()


@attr.s
class RunResult:
    moving: np.ndarray = attr.ib()
    static: np.ndarray = attr.ib()

    centroids: np.ndarray = attr.ib()
    intervals: np.ndarray = attr.ib()

    results: list = attr.ib()
    warped_moving: list = attr.ib()


class StreamlitProgressWrapper:
    def __init__(self, total):
        self.total = total
        self.n = 0
        self.pbar = st.progress(0.)

    def update(self, delta):
        self.n += delta
        self.pbar.progress(self.n / self.total)

    def set_postfix(self, *args, **kwargs):
        print("set_postfix currently unsupported.")



RUNS_DIRECTORY = "data/runs"
CONFIG_SUFFIX = ".config"
RESULTS_SUFFIX = ".results"

# load previous configs
configs = []
config_paths = [p for p in os.listdir(RUNS_DIRECTORY) if p.endswith(CONFIG_SUFFIX)]
for fp in config_paths:
    with open(os.path.join(RUNS_DIRECTORY, fp), 'rb') as config_file:
        configs.append(pickle.load(config_file))


def figure_to_image():
    canvas = plt.gcf().canvas
    canvas.draw()
    buf = canvas.buffer_rgba()
    plt.close()
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

'''
# Contour-based registration
### feature map
'''

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


@st.cache(show_spinner=False)
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


@st.cache(allow_output_mutation=True)
def plot_colored_feature_map():
    colored_map = 1 - np.tile(feature_map[..., None], reps=(1, 1, 3)) * 0.3
    colored_map[patch_slice[0], patch_slice[1], :] = get_colored_difference_image(moving=feature_map[patch_slice])
    colored_map[window_slice[0], window_slice[1], :] = get_colored_difference_image(static=feature_map[window_slice])
    colored_map[intersection_slice[0], intersection_slice[1], :] = get_colored_difference_image(
        static=feature_map[intersection_slice],
        moving=feature_map[intersection_slice])
    plt.imshow(colored_map)
    return figure_to_image()


feature_map_plot_placeholder.image(image=plot_colored_feature_map())

'''
This is an overlay of the two chosen patches.
'''


@st.cache(allow_output_mutation=True)
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
centroids_degrees = [f"{-(c / np.pi * 180 + 180) % 360:.0f}°" for c in centroids]
centroids_colors = angle_to_rgb(centroids)


def write_centroid_legend():
    centroid_colors = [matplotlib.colors.to_hex(c) for c in centroids_colors]
    items = [f'<span style="color:{c}">{cd}</span>' for c, cd in zip(centroid_colors, centroids_degrees)]
    html = '<div style="background-color: black; color:white; float:left">Angles: ' + ', '.join(items) + '</div>'
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


@st.cache(allow_output_mutation=True)
def plot_angles():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')

    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks(centroids)
    ax.set_xticklabels(centroids_degrees)

    for angle, (ilow, ihigh), color in zip(centroids, intervals, centroids_colors):
        ax.plot([angle, angle], [0, 1], c=color)
        if ilow > ihigh:
            ihigh += 2 * np.pi
        ax.fill_between(np.linspace(ilow, ihigh, num=3), 0, 2, color=color, alpha=0.1)

    ax.set_ylim(0, 1)
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


@st.cache(allow_output_mutation=True)
def plot_assignments():
    _, axs = plt.subplots(1, 2, figsize=(8, 4))
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


@st.cache(allow_output_mutation=True)
def plot_memberships():
    plt.figure()
    memberships = get_memberships_from_centroids(moving, centroids, intervals)
    color = centroids_colors[picked_angle_index]
    membership_image = np.tile(moving[..., None], reps=(1, 1, 3)) * 0.2
    membership_image += 0.8 * memberships[picked_angle_index, ..., None] * color
    plt.imshow(membership_image)
    return figure_to_image()


st.image(image=plot_memberships())

# TODO plot assignments

'''
### Type of transform
Based on the directions, we can now fit a transformation.
'''

transform_type = st.radio(label='transform type', options=("linear transform",
                                                           "dense displacement"))

if transform_type == 'dense displacement':
    assignment_type = st.radio('assignment type', options=("binary", "memberships"))
    smoothness = st.slider('warp field smoothness', min_value=0, max_value=10000, value=2000)
else:
    smoothness = None

num_iterations = st.number_input('number of iterations', min_value=1, max_value=100, value=20)


'''
### Results
'''


config = RunConfiguration(patch_position, centroid_method, num_centroids, kde_rho, transform_type,
                          smoothness, num_iterations)

def load_config_and_show():
    config_path = config_paths[configs.index(config)].replace(CONFIG_SUFFIX, RESULTS_SUFFIX)
    with open(os.path.join(RUNS_DIRECTORY, config_path), 'rb') as results_file:
        run_result: RunResult = pickle.load(results_file)

    result_index_placeholder = st.empty()
    result_index = result_index_placeholder.slider(label="Pick frame", min_value=0,
                                                   max_value=num_iterations,
                                                   value=num_iterations, step=1,
                                                   key="result_index_slider_initial")
    animate_button = st.button(label='Animate')

    @st.cache(allow_output_mutation=True)
    def show_result(i):
        if i == 0:
            plot_diff(moving, static)
        else:
            plot_diff(run_result.warped_moving[i - 1], static)
        return figure_to_image()

    result_diff_placeholder = st.empty()

    if animate_button:
        for i in range(num_iterations + 1):
            result_diff_placeholder.image(image=show_result(i), use_column_width=True)
            result_index_placeholder.slider(label="Animating...", min_value=0, max_value=num_iterations,
                                            value=i, step=1)
            time.sleep(0.5)
        result_index = result_index_placeholder.slider(label="Pick frame", min_value=0,
                                                       max_value=num_iterations,
                                                       value=num_iterations, step=1,
                                                       key="result_index_slider_post_animate")

    result_diff_placeholder.image(image=show_result(result_index), use_column_width=True)


if config in configs:
    load_config_and_show()

elif st.button("Run calculation with above settings"):
    pbar = StreamlitProgressWrapper(total=num_iterations)

    random_name = ''.join(random.choices(string.ascii_lowercase, k=16))
    with open(os.path.join(RUNS_DIRECTORY, random_name+CONFIG_SUFFIX), 'wb') as config_file:
        pickle.dump(config, config_file)

    result_obj = RunResult(moving.copy(), static.copy(), centroids.copy(), intervals.copy(), None, None)
    # run calculation
    results = None
    if transform_type == 'linear transform':
        results = estimate_transform_from_correspondences(moving, static, num_iterations, centroids, intervals, pbar)
    elif transform_type == 'dense displacement':
        # TODO binary/membership radio does not do anything yet
        results = estimate_dense_displacements_from_memberships(moving, static, num_iterations, centroids, intervals, smoothness, pbar)


    result_obj.results = results
    result_obj.warped_moving = [apply_transform(moving, r.stacked_transform) for r in results]

    with open(os.path.join(RUNS_DIRECTORY, random_name+RESULTS_SUFFIX), 'wb') as results_file:
        pickle.dump(result_obj, results_file)

    config_paths.append(random_name+CONFIG_SUFFIX)
    configs.append(config)

    load_config_and_show()

