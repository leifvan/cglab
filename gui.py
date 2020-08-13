import random
import string
import time
from functools import partial
from pathlib import Path

import altair as alt
import attr
import imageio
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.transform
import streamlit as st

import gui_config as conf
from displacement import plot_correspondences, get_correspondences_energy, plot_projective_transform
from distance_transform import get_binary_assignments_from_centroids, get_distance_transforms_from_binary_assignments, \
    get_closest_feature_directions_from_binary_assignments, get_memberships_from_centroids, \
    get_binary_assignments_from_gabor, get_memberships_from_gabor
from gradient_directions import get_n_equidistant_angles_and_intervals, get_main_gradient_angles_and_intervals, \
    plot_gradients_as_arrows, wrapped_cauchy_kernel_density, get_gradients_in_polar_coords, plot_binary_assignments, \
    plot_gabor_filter
from gui_utils import figure_to_image, load_previous_configs, RunConfiguration, CONFIG_SUFFIX, RUNS_DIRECTORY, \
    RunResult, StreamlitProgressWrapper, PartialRunConfiguration, make_st_widget
from methods import apply_transform, estimate_linear_transform, estimate_dense_displacements
from patches import find_promising_patch_pairs
from utils import plot_diff, pad_slices, get_colored_difference_image, get_slice_intersection, angle_to_rgb

cache_allow_output_mutation = partial(st.cache, allow_output_mutation=True)

configs = load_previous_configs()
params = PartialRunConfiguration()

'''
# Gradient-based registration
'''

config = conf.DEFAULT_CONFIG
config_name = st.sidebar.text_input("load an existing config", max_chars=16, value="")
if st.sidebar.button("load config"):
    config_name_to_config = {c.name: c for c in configs}
    if config_name in config_name_to_config:
        config = config_name_to_config[config_name]
        st.sidebar.success("Config loaded.")
    else:
        st.sidebar.error("Config not found.")

st.sidebar.markdown("---")

feature_map_paths = conf.FEATURE_MAP_DIR.glob("*.png")
params.feature_map_path = st.sidebar.selectbox("Choose a feature map",
                                               options=[p.name for p in feature_map_paths])

params.downscale_factor = make_st_widget(conf.DOWNSCALE_FACTOR_DESCRIPTOR,
                                         label="downscale factor",
                                         value=config.downscale_factor)


@cache_allow_output_mutation
def get_feature_map():
    feature_map = imageio.imread(conf.FEATURE_MAP_DIR / params.feature_map_path).astype(np.float32)

    if len(feature_map.shape) == 3:
        feature_map = np.mean(feature_map, axis=2)

    feature_map = skimage.transform.downscale_local_mean(feature_map, (params.downscale_factor,
                                                                       params.downscale_factor))
    feature_map /= feature_map.max()
    feature_map[feature_map > 0.5] = 1
    feature_map[feature_map < 0.5] = 0
    return feature_map


feature_map = get_feature_map()

'''
### patch pair
'''

st.markdown('Here we select two patches that approximately match. The moving patch is colored in '
            '<span style="color:indianred">**red**</span>, the static patch is '
            '<span style="color:steelblue">**blue**</span>, and their intersection is '
            '<span style="color:gold">**yellow**</span> (if applicable).',
            unsafe_allow_html=True)

feature_map_plot_placeholder = st.empty()

f'''
### patch selection
A simple patch matching algorithm is run on the image to find two similar patches. Similarity is
measured as the MAE between the images. The index determines which of the
{conf.NUM_PATCH_PAIRS} best pairs to pick. The similarity decreases with higher values.
'''

params.patch_position = make_st_widget(conf.PATCH_POSITION_DESCRIPTOR,
                                       label="index of the patch pair to choose",
                                       value=config.patch_position)

st.sidebar.markdown('---')


@cache_allow_output_mutation(show_spinner=False, persist=True)
def get_patch_pairs():
    return find_promising_patch_pairs(feature_map, patch_size=conf.PATCH_SIZE,
                                      stride=64 // params.downscale_factor,
                                      padding=conf.PADDING_SIZE,
                                      num_pairs=conf.NUM_PATCH_PAIRS)


patch_pairs = get_patch_pairs()
patch_slice, window_slice, _ = patch_pairs[conf.NUM_PATCH_PAIRS - params.patch_position]


@cache_allow_output_mutation
def get_moving_and_static():
    padded_window_slice = pad_slices(window_slice, padding=conf.PADDING_SIZE,
                                     assert_shape=feature_map.shape)
    feature_patch = np.pad(feature_map[patch_slice], conf.PADDING_SIZE)
    feature_window = feature_map[padded_window_slice]
    return feature_patch, feature_window


moving, static = get_moving_and_static()
intersection_slice = get_slice_intersection(patch_slice, window_slice)
IMAGE_FOR_MAIN_DIRECTIONS = static


@cache_allow_output_mutation
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


@cache_allow_output_mutation
def plot_moving_static_diff():
    plot_diff(moving, static)
    return figure_to_image()


st.image(image=plot_moving_static_diff(), use_column_width=True)

'''
### Finding dominant gradient directions
Now we are looking for the main directions of gradients in the (static) image. **The following
statement is not true for Gabor** For each main
direction we also need an interval s.t. every angle in that interval is assigned to the main
direction.
'''

params.centroid_method = make_st_widget(conf.CENTROID_METHOD_DESCRIPTOR,
                                        label="how to determine main gradient directions",
                                        value=config.centroid_method)

if params.centroid_method == conf.CentroidMethod.EQUIDISTANT:
    '''
    Here we simply choose $n$ equidistant directions and intervals.
    '''
    params.num_centroids = make_st_widget(conf.NUM_CENTROIDS_DESCRIPTOR,
                                          label="number of equidistant angles",
                                          value=config.num_centroids)
elif params.centroid_method == conf.CentroidMethod.HISTOGRAM_CLUSTERING:
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
    params.kde_rho = make_st_widget(conf.KDE_RHO_DESCRIPTOR,
                                    label="rho-value for the KDE",
                                    value=config.kde_rho)
else:
    raise AttributeError(f"Centroid method incorrect. {params}")

st.sidebar.markdown('---')


@cache_allow_output_mutation
def get_centroids_intervals():
    if params.centroid_method == conf.CentroidMethod.EQUIDISTANT:
        return get_n_equidistant_angles_and_intervals(params.num_centroids)
    elif params.centroid_method == conf.CentroidMethod.HISTOGRAM_CLUSTERING:
        return get_main_gradient_angles_and_intervals(IMAGE_FOR_MAIN_DIRECTIONS, params.kde_rho)
    raise AttributeError(params)


centroids, intervals = get_centroids_intervals()
centroids_degrees = [f"{-(c / np.pi * 180 + 180) % 360:.0f}°" for c in centroids]
centroids_colors = angle_to_rgb(centroids)


def write_centroid_legend():
    centroid_colors = [matplotlib.colors.to_hex(c) for c in centroids_colors]
    items = [f'<span style="color:{c}">{cd}</span>' for c, cd in zip(centroid_colors, centroids_degrees)]
    html = '<div style="background-color: black; color:white; float:left">Angles: ' + ', '.join(items) + '</div>'
    st.markdown(html, unsafe_allow_html=True)


@cache_allow_output_mutation
def get_kde_plot_data():
    # TODO this is just copy-and-pasted from the code in gradient_directions.py
    angles, magnitudes = get_gradients_in_polar_coords(IMAGE_FOR_MAIN_DIRECTIONS)

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
                                           rho=params.kde_rho)
    return sample_points, scores


@cache_allow_output_mutation
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

    if params.centroid_method == 'histogram clustering':
        sample_points, scores = get_kde_plot_data()
        ax.plot(sample_points, scores)
        ax.set_ylim(0, scores.max())

    return figure_to_image()


''
st.image(image=plot_angles())

'''
### Binary assignments
'''

params.filter_method = make_st_widget(conf.FILTER_METHOD_DESCRIPTOR,
                                      label="method for retrieving angle responses",
                                      value=config.filter_method)
if params.filter_method == conf.FilterMethod.FARID_DERIVATIVE:
    get_binary_assignments = get_binary_assignments_from_centroids
    get_memberships = get_memberships_from_centroids
elif params.filter_method == conf.FilterMethod.GABOR:
    params.gabor_filter_sigma = make_st_widget(conf.GABOR_FILTER_SIGMA_DESCRIPTOR,
                                               label="gabor filter sigma",
                                               value=config.gabor_filter_sigma)
    get_binary_assignments = partial(get_binary_assignments_from_gabor, sigma=params.gabor_filter_sigma)
    get_memberships = partial(get_memberships_from_gabor, sigma=params.gabor_filter_sigma)

params.response_cutoff_threshold = make_st_widget(conf.RESPONSE_CUTOFF_THRESHOLD,
                                                  label="response cutoff")

centroids_degrees_and_all = ('-- all --', *centroids_degrees)
picked_angle = st.selectbox(label='angle for assignments', options=centroids_degrees_and_all)
picked_angle_index = centroids_degrees_and_all.index(picked_angle) - 1

write_centroid_legend()


@cache_allow_output_mutation
def plot_assignments():
    # _, axs = plt.subplots(1, 2, figsize=(8, 4))
    moving_assignments = get_binary_assignments(moving, centroids, intervals,
                                                threshold=params.response_cutoff_threshold)
    static_assignments = get_binary_assignments(static, centroids, intervals,
                                                threshold=params.response_cutoff_threshold)

    if picked_angle_index == -1:
        plt.figure(figsize=(8, 4))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        plot_binary_assignments(moving_assignments, centroids, ax1)
        plot_binary_assignments(static_assignments, centroids, ax2)
    else:
        plt.figure(figsize=(8, 8))
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        ax4 = plt.subplot(2, 2, 4)

        i = picked_angle_index
        plot_binary_assignments(moving_assignments[i, None], centroids[i, None], ax1)
        plot_binary_assignments(static_assignments[i, None], centroids[i, None], ax2)
        plot_gabor_filter(centroids[i], params.gabor_filter_sigma, ax=ax3)
        ax3.set_title("Gabor filter")
        distances = get_distance_transforms_from_binary_assignments(static_assignments[i,None])
        ax4.imshow(distances[0], cmap='bone')

    ax1.set_title("moving")
    ax2.set_title("static")


    return figure_to_image()


st.image(image=plot_assignments(), use_column_width=True)


@cache_allow_output_mutation
def get_moving_assignments_memberships():
    assignments = get_binary_assignments(moving, centroids, intervals, threshold=params.response_cutoff_threshold)
    memberships = get_memberships(moving, centroids, intervals, threshold=params.response_cutoff_threshold)
    return assignments, memberships


moving_assignments, moving_memberships = get_moving_assignments_memberships()


@cache_allow_output_mutation
def get_static_assignments_distances_directions():
    assignments = get_binary_assignments(static, centroids, intervals, threshold=params.response_cutoff_threshold)
    distances = get_distance_transforms_from_binary_assignments(assignments)
    directions = get_closest_feature_directions_from_binary_assignments(assignments)
    # directions = get_closest_feature_directions_from_distance_transforms(distances)
    return assignments, distances, directions


static_assignments, static_distances, static_directions = get_static_assignments_distances_directions()

params.assignment_type = make_st_widget(conf.ASSIGNMENT_TYPE_DESCRIPTOR,
                                        label="assignment type",
                                        value=config.assignment_type)

if params.assignment_type == conf.AssignmentType.MEMBERSHIPS:

    r'''
    ### Memberships
    Instead of using binary assignments in the moving image, we will use soft-assignments for each
    (pixel, angle) pair. Let $M_{\phi_k}[i,j]$ be the membership of pixel (i,j) to the k-th main
    direction $\phi_k$. The membership is a value in [0,1] and is determined as a linear
    interpolation where $M_\phi[i,j]$ is 0 if the angle $\psi$ at pixel (i,j) is outside the
    interval $(\phi_{k-1}, \phi_{k+1})$ and 1 if $\psi = \phi_k$. These values are shown in the
    following plot:
    
    '''


    @cache_allow_output_mutation
    def plot_membership_calculation():
        plt.figure(figsize=(7, 3))
        centroids_degree_values = -(centroids / np.pi * 180 + 180) % 360
        sort_idx = np.argsort(centroids_degree_values)
        centroids_degree_values = centroids_degree_values[sort_idx]

        for i, color in enumerate(centroids_colors[sort_idx]):
            if i == 0:
                plt.plot(centroids_degree_values[:2], [1, 0], c=color)
                plt.fill_between(centroids_degree_values[:2], [1, 0], color=color, alpha=0.05)
            elif i == len(centroids_colors) - 1:
                plt.plot([centroids_degree_values[i - 1],
                          centroids_degree_values[i],
                          centroids_degree_values[0] + 360], [0, 1, 0], c=color)
                plt.fill_between([centroids_degree_values[i - 1],
                                  centroids_degree_values[i],
                                  centroids_degree_values[0] + 360], [0, 1, 0],
                                 color=color, alpha=0.05)
            else:
                plt.plot(centroids_degree_values[i - 1:i + 2], [0, 1, 0], c=color)
                plt.fill_between(centroids_degree_values[i - 1:i + 2], [0, 1, 0], color=color, alpha=0.05)

        plt.plot([centroids_degree_values[-1], centroids_degree_values[0] + 360], [0, 1],
                 c=centroids_colors[sort_idx[0]])
        plt.fill_between([centroids_degree_values[-1], centroids_degree_values[0] + 360], [0, 1],
                         color=centroids_colors[sort_idx[0]], alpha=0.05)
        plt.xticks([*centroids_degree_values, centroids_degree_values[0] + 360],
                   [f"{cdv:.0f}°" for cdv in [*centroids_degree_values, centroids_degree_values[0] + 360]])
        plt.xlabel("angle")
        plt.ylabel("membership")
        plt.tight_layout()
        return figure_to_image()


    st.image(image=plot_membership_calculation())

    '''
    This gives the following memberships for each pixel of the moving image:
    '''

    picked_angle = st.selectbox(label='angle', options=centroids_degrees)
    picked_angle_index = centroids_degrees.index(picked_angle)

    write_centroid_legend()


    @cache_allow_output_mutation
    def plot_memberships():
        plt.figure()
        color = centroids_colors[picked_angle_index]
        membership_image = np.tile(moving[..., None], reps=(1, 1, 3)) * 0.2
        membership_image += 0.8 * moving_memberships[picked_angle_index, ..., None] * color
        plt.imshow(membership_image)
        return figure_to_image()


    st.image(image=plot_memberships())

params.weight_correspondence_angles = make_st_widget(conf.WEIGHT_CORRESPONDENCE_ANGLES_DESCRIPTOR,
                                                     label="weight correspondences on similarity with main direction")

'''
### Fit a transformation
'''

params.transform_type = make_st_widget(conf.TRANSFORM_TYPE_DESCRIPTOR,
                                       label="transform type",
                                       value=config.transform_type)
transform_type_to_name = {conf.TransformType.LINEAR: 'projective transformation',
                          conf.TransformType.DENSE: 'dense displacement map'}

f'''
We can now fit a *{transform_type_to_name[params.transform_type]}*.
'''

r'''
The transform will be fitted based on the following correspondences. The correspondences
are determined using the distance transformations of the binary map of each main angle.
More specifically, for a main direction $\phi$ and its interval $[a, b]$, we get two
binary images $M_\phi$ and $S_\phi$ indicating the pixels with a gradient
angle in $[a,b]$ for the moving and the static image, respectively. For every pixel
$(i,j)$ where $M_\phi[i,j] = 1$ we can find a closest pixel $(i',j')$ with
$S_\phi[i',j'] = 1$. These pairs are the correspondences.

Below all correspondences are drawn as arrows from pixel $(i,j)$ in $M$ to the pixel
$(i',j')$ in $S$. The coloring denotes from which main direction $\phi$ the
correspondence originates.
'''

if params.assignment_type == conf.AssignmentType.MEMBERSHIPS:
    '''
    The correspondences will be weighted by the membership values (see above). In the following
    plot, the weights are depicted by the transparency of the arrows.
    '''

picked_angle = st.selectbox(label="Choose specific angle", options=centroids_degrees_and_all)
picked_angle_index = centroids_degrees_and_all.index(picked_angle) - 1

write_centroid_legend()


#@cache_allow_output_mutation
def plot_binary_correspondences():
    plt.figure()
    if params.assignment_type == conf.AssignmentType.BINARY:
        memberships = moving_assignments
    elif params.assignment_type == conf.AssignmentType.MEMBERSHIPS:
        memberships = moving_memberships

    if picked_angle_index == -1:
        plot_correspondences(moving, static, centroids, memberships,
                             static_distances, static_directions,
                             weight_correspondence_angles=params.weight_correspondence_angles)
    else:
        plot_correspondences(moving, static, centroids[picked_angle_index, None],
                             memberships[picked_angle_index, None],
                             static_distances[picked_angle_index, None],
                             static_directions[picked_angle_index, None],
                             weight_correspondence_angles=params.weight_correspondence_angles)
    plt.title("correspondences from " +
              ("binary assignments" if params.assignment_type == 'binary' else "memberships"))
    plt.tight_layout()
    return figure_to_image()


st.image(plot_binary_correspondences())

if params.transform_type == conf.TransformType.LINEAR:

    r'''
    The projective transform
    $$
    \begin{pmatrix}
    m_1 & m_2 & t_1 \\
    m_3 & m_4 & t_2 \\
    c_1 & c_2 & 1
    \end{pmatrix}
    $$
    induces the following system of linear equations
    $$
    \begin{pmatrix}
    \vdots \\
    \begin{matrix}
        i & j & 1 & 0 & 0 & 0 & -i\cdot i' & -j\cdot i' \\
        0 & 0 & 0 & i & j & 1 & -i\cdot j' & -j\cdot j'
    \end{matrix} \\
    \vdots
    \end{pmatrix}
    \begin{pmatrix}
    m_1 \\ m_2 \\ t_1 \\ m_3 \\ m_4 \\ t_2 \\ c_1 \\ c_2 
    \end{pmatrix} = 
    \begin{pmatrix}
        i' \\ j'
    \end{pmatrix}
    $$
    '''
    params.l2_regularization_factor = make_st_widget(conf.L2_REGULARIZATION_FACTOR_DESCRIPTOR,
                                                     label="L2 regularization",
                                                     value=config.l2_regularization_factor)

elif params.transform_type == conf.TransformType.DENSE:
    r'''
    The dense transformation will be fitted using radial basis function (RBF) interpolation. This
    means the x- and y-offsets $\Delta x(i,j)$ and $\Delta y(i,j)$ for each pixel $(i,j)$ will be
    interpolated between the offsets induced by the correspondences. Let $\Delta(i,j)$ be one of
    the offsets. Then the RBF interpolation for a RBF $\varphi(r)$ is given as
    $$
    \Delta(i,j) = \sum_{k=1}^n w_k\cdot\varphi\left(\left\lVert \begin{pmatrix}i\\j\end{pmatrix} - 
        \begin{pmatrix} i_k \\ j_k \end{pmatrix} \right\rVert\right)
    $$ 
    where 
    '''
    params.rbf_type = make_st_widget(conf.RBF_TYPE_DESCRIPTOR, label="RBF type", value=config.rbf_type)
    params.smoothness = make_st_widget(conf.SMOOTHNESS_DESCRIPTOR,
                                       label="warp field smoothness",
                                       value=config.smoothness)
    params.num_dct_coeffs = make_st_widget(conf.NUM_DCT_COEFFS_DESCRIPTOR,
                                           label="spectral coefficients",
                                           value=config.num_dct_coeffs)
    if params.num_dct_coeffs == moving.shape[0]:
        params.num_dct_coeffs = None

transform_dof = None
if params.transform_type == conf.TransformType.LINEAR:
    transform_dof = 8
elif params.transform_type == conf.TransformType.DENSE:
    if params.num_dct_coeffs is None or params.num_dct_coeffs == 0:
        transform_dof = 2 * moving.size
    else:
        transform_dof = 2 * (params.num_dct_coeffs ** 2)

st.sidebar.markdown(f"The resulting transform has {transform_dof} degrees of freedom.")
params.num_iterations = make_st_widget(conf.NUM_ITERATIONS_DESCRIPTOR,
                                       label="number of iterations",
                                       value=config.num_iterations)

'''
## Results
'''

random_name = Path(''.join(random.choices(string.ascii_lowercase, k=16)) + CONFIG_SUFFIX)
params.file_path = RUNS_DIRECTORY / random_name
config = RunConfiguration(**attr.asdict(params))


def load_config_and_show():
    run_result = config.load_results()

    result_index_placeholder = st.empty()
    result_index = result_index_placeholder.slider(label="Pick frame", min_value=0,
                                                   max_value=config.num_iterations,
                                                   value=config.num_iterations, step=1,
                                                   key="result_index_slider_initial")

    run_animate = st.button("Animate")

    @cache_allow_output_mutation
    def show_result(i):
        _, axs = plt.subplots(2, 3, figsize=(12, 10))

        warped_moving = moving if i == 0 else run_result.warped_moving[i - 1]
        axs[0, 0].imshow(get_colored_difference_image(moving, static))
        axs[0, 0].set_title("unwarped")

        if i > 0:
            if params.transform_type == 'linear transform':
                plot_projective_transform(run_result.results[i - 1].stacked_transform, ax=axs[0, 1])
            elif params.transform_type == 'dense displacement':
                local_transform = run_result.results[i - 1].stacked_transform - np.mgrid[:moving.shape[0],
                                                                                :moving.shape[1]]
                plot_gradients_as_arrows(*local_transform, subsample=4, ax=axs[0, 1])
        axs[0, 1].set_title("estimated transform")

        axs[0, 2].imshow(get_colored_difference_image(warped_moving, static))
        axs[0, 2].set_title("warped")

        # SECOND ROW
        warped_moving_assignments = get_binary_assignments(warped_moving, centroids, intervals,
                                                           threshold=config.response_cutoff_threshold)

        plot_binary_assignments(static_assignments, centroids, ax=axs[1, 0])
        axs[1, 0].set_title("static assignments")

        plot_correspondences(warped_moving, static, centroids, warped_moving_assignments,
                             static_distances, static_directions,
                             weight_correspondence_angles=params.weight_correspondence_angles,
                             ax=axs[1, 1])
        axs[1, 1].set_title("warped correspondences")
        # if config.assignment_type == conf.AssignmentType.BINARY:

        plot_binary_assignments(warped_moving_assignments, centroids, ax=axs[1, 2])
        # TODO plot memberships somehow
        # elif config.assignment_type == conf.AssignmentType.MEMBERSHIPS:
        #     warped_moving_memberships = get_memberships(warped_moving, centroids, intervals,
        #                                                 threshold=config.response_cutoff_threshold)
        #     plot_memberships()
        axs[1, 2].set_title("warped moving assignments")

        plt.tight_layout()
        return figure_to_image()

    result_diff_placeholder = st.empty()

    if run_animate:
        for i in range(0, config.num_iterations + 1, 1 + config.num_iterations // 50):
            start_time = time.time()
            result_diff_placeholder.image(image=show_result(i), use_column_width=True)
            result_index_placeholder.slider(label="Animating...", min_value=0,
                                            max_value=config.num_iterations,
                                            value=i, step=1)
            time.sleep(max(0.5 - time.time() + start_time, 0))
        result_index = result_index_placeholder.slider(label="Pick frame", min_value=0,
                                                       max_value=config.num_iterations,
                                                       value=config.num_iterations, step=1,
                                                       key="result_index_slider_post_animate")

    result_diff_placeholder.image(image=show_result(result_index), use_column_width=True)

    similar_configs = [c for c in configs if c.is_similar_to(config)]

    '''
    #### Energy
    The energy is determined as the sum of the distances of the correspondences, weighted by the membership
    and normalized by the sum of memberships.
    '''
    initial_energy = get_correspondences_energy(moving_memberships, static_distances)

    if len(similar_configs) > 1 and st.checkbox(f'Show energy for {len(similar_configs) - 1} similar configs'):

        filter_names = [a.name for a in attr.fields(RunConfiguration) if a.eq and a.name != "feature_map_path"]
        filters = st.multiselect('What values should be equal?', options=filter_names)

        df = pd.DataFrame([attr.asdict(c) for c in similar_configs])

        if len(filters) > 0:
            df = df.query('&'.join(f"{f}==@config.{f}" for f in filters))

        filtered_paths = [Path(p) for p in df['file_path']]
        similar_results = [c.load_results() for c in similar_configs if c.file_path in filtered_paths]

        energies = [[initial_energy] + [r.energy for r in rr.results] for rr in similar_results]
        all_energy = np.concatenate([[[i, j, e] for j, e in enumerate(energy)]
                                     for i, energy in enumerate(energies)],
                                    axis=0)

        df.reset_index(inplace=True, drop=True)
        all_energy_df = pd.DataFrame(all_energy, columns=['id', 'x', 'energy']).join(df, on='id')

        df = df.drop(columns=['file_path', 'patch_position'])
        st.dataframe(df)

        alt_chart = alt.Chart(all_energy_df) \
            .mark_line(point=True) \
            .encode(x='x:Q', y='energy:Q', color='id:N',
                    tooltip=['id'] + list(df.columns)).interactive()
        st.altair_chart(alt_chart, use_container_width=True)
    else:
        energy_df = pd.DataFrame({
            'iteration': list(range(len(run_result.results))),
            'energy': [r.energy for r in run_result.results]})
        altair_chart = alt.Chart(energy_df).mark_line().encode(x='iteration', y='energy', )
        st.altair_chart(altair_chart, use_container_width=True)


if config in configs:
    config = configs[configs.index(config)]
    st.sidebar.text("Calculation done.")
    if config.is_favorite:
        st.sidebar.markdown("*Config is saved as favorite.*")
    elif st.sidebar.button("Save as favorite"):
        config = RunConfiguration(**{**attr.asdict(config), 'is_favorite': True})
        config.save()

    load_config_and_show()

elif st.sidebar.button("Run calculation"):

    pbar = StreamlitProgressWrapper(total=config.num_iterations)

    # run calculation
    # TODO improve this
    # TODO maybe recompute centroids, intervals etc.?
    results = None
    assignment_fn = None

    if config.assignment_type == 'binary' and config.filter_method == 'Farid derivative filter':
        assignment_fn = get_binary_assignments_from_centroids
    elif config.assignment_type == 'binary' and config.filter_method == 'Gabor filter':
        assignment_fn = partial(get_binary_assignments_from_gabor, sigma=config.gabor_filter_sigma)
    elif config.assignment_type == 'memberships' and config.filter_method == 'Farid derivative filter':
        assignment_fn = get_memberships_from_centroids
    elif config.assignment_type == 'memberships' and config.filter_method == 'Gabor filter':
        assignment_fn = partial(get_memberships_from_gabor, sigma=config.gabor_filter_sigma)

    common_params = dict(moving=moving, static=static, n_iter=config.num_iterations,
                         centroids=centroids, intervals=intervals, progress_bar=pbar,
                         assignments_fn=assignment_fn,
                         weight_correspondence_angles=config.weight_correspondence_angles)

    estimate_fn = None
    if config.transform_type == 'linear transform':
        estimate_fn = partial(estimate_linear_transform, reg_factor=config.l2_regularization_factor)
    elif config.transform_type == 'dense displacement':
        estimate_fn = partial(estimate_dense_displacements, smooth=config.smoothness,
                              rbf_type=config.rbf_type, reduce_coeffs=config.num_dct_coeffs)

    results = estimate_fn(**common_params)

    if results is None:
        st.error("Failed to run config!")
    else:
        config.save()

        result_obj = RunResult(moving=moving.copy(), static=static.copy(), centroids=centroids.copy(),
                               intervals=intervals.copy(), results=results,
                               warped_moving=[apply_transform(moving, r.stacked_transform) for r in results])
        config.save_results(result_obj)
        configs.append(config)

        load_config_and_show()
else:
    st.info("Click 'Run calculation' in the sidebar to get results.")
