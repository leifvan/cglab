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
from blending import histogram_preserved_blending
from displacement import plot_correspondences, get_correspondences_energy, plot_projective_transform
from distance_transform import get_binary_assignments_from_centroids, get_distance_transforms_from_binary_assignments, \
    get_closest_feature_directions_from_binary_assignments, get_memberships_from_centroids, \
    get_binary_assignments_from_gabor, get_memberships_from_gabor
from gradient_directions import get_n_equidistant_angles_and_intervals, get_main_gradient_angles_and_intervals, \
    plot_gradients_as_arrows, wrapped_cauchy_kernel_density, get_gradients_in_polar_coords, plot_binary_assignments
from gui_config import RunResult, PartialRunConfiguration, make_st_widget
from gui_plotting import plot_centroids_intervals_polar, plot_multiple_binary_assignments, plot_memberships
from gui_utils import figure_to_image, load_previous_configs, RunConfiguration, CONFIG_SUFFIX, RUNS_DIRECTORY, \
    StreamlitProgressWrapper, load_and_preprocess_feature_map, get_padded_moving_and_static, run_config
from methods import apply_transform, estimate_linear_transform, estimate_dense_displacements
from patches import find_promising_patch_pairs
from utils import plot_diff, pad_slices, get_colored_difference_image, get_slice_intersection, angle_to_rgb

cache_allow_output_mutation = partial(st.cache, allow_output_mutation=True)

configs = load_previous_configs()
params = PartialRunConfiguration()

# get data for sidebar
config = conf.DEFAULT_CONFIG

feature_map_paths = list(conf.FEATURE_MAP_DIR.glob("*.png"))
texture_path_dict = {p.name: conf.TEXTURE_DIR / p.name for p in feature_map_paths}

# ++++++++++++++
#    SIDEBAR
# ++++++++++++++

config_name = st.sidebar.text_input("load an existing config", max_chars=16, value="")
if st.sidebar.button("load config"):
    config_name_to_config = {c.name: c for c in configs}
    if config_name in config_name_to_config:
        config = config_name_to_config[config_name]
        st.sidebar.success("Config loaded.")
    else:
        st.sidebar.error("Config not found.")

st.sidebar.markdown("---")

params.feature_map_path = st.sidebar.selectbox("Choose a feature map",
                                               options=[p.name for p in feature_map_paths])

params.downscale_factor = make_st_widget(conf.DOWNSCALE_FACTOR_DESCRIPTOR,
                                         label="downscale factor",
                                         value=config.downscale_factor)

# FIXME this is now hacked, because this just generates origins, see below
params.patch_position = make_st_widget(conf.PATCH_POSITION_DESCRIPTOR,
                                       label="index of the patch pair to choose",
                                       value=config.patch_position)

params.centroid_method = make_st_widget(conf.CENTROID_METHOD_DESCRIPTOR,
                                        label="how to determine main gradient directions",
                                        value=config.centroid_method)

if params.centroid_method == conf.CentroidMethod.EQUIDISTANT:
    params.num_centroids = make_st_widget(conf.NUM_CENTROIDS_DESCRIPTOR,
                                          label="number of equidistant angles",
                                          value=config.num_centroids)
elif params.centroid_method == conf.CentroidMethod.HISTOGRAM_CLUSTERING:
    params.kde_rho = make_st_widget(conf.KDE_RHO_DESCRIPTOR,
                                    label="rho-value for the KDE",
                                    value=config.kde_rho)

st.sidebar.markdown('---')

params.filter_method = make_st_widget(conf.FILTER_METHOD_DESCRIPTOR,
                                      label="method for retrieving angle responses",
                                      value=config.filter_method)

if params.filter_method == conf.FilterMethod.GABOR:
    params.gabor_filter_sigma = make_st_widget(conf.GABOR_FILTER_SIGMA_DESCRIPTOR,
                                               label="gabor filter sigma",
                                               value=config.gabor_filter_sigma)

params.response_cutoff_threshold = make_st_widget(conf.RESPONSE_CUTOFF_THRESHOLD,
                                                  label="response cutoff")

st.sidebar.markdown('---')

params.assignment_type = make_st_widget(conf.ASSIGNMENT_TYPE_DESCRIPTOR,
                                        label="assignment type",
                                        value=config.assignment_type)

params.weight_correspondence_angles = make_st_widget(conf.WEIGHT_CORRESPONDENCE_ANGLES_DESCRIPTOR,
                                                     label="weight correspondences on similarity with main direction")

st.sidebar.markdown('---')

params.transform_type = make_st_widget(conf.TRANSFORM_TYPE_DESCRIPTOR,
                                       label="transform type",
                                       value=config.transform_type)

if params.transform_type == conf.TransformType.LINEAR:
    params.linear_transform_type = make_st_widget(conf.LINEAR_TRANSFORM_TYPE_DESCRIPTOR,
                                                  label='linear transform type',
                                                  value=config.linear_transform_type)

    params.l2_regularization_factor = make_st_widget(conf.L2_REGULARIZATION_FACTOR_DESCRIPTOR,
                                                     label="L2 regularization",
                                                     value=config.l2_regularization_factor)
elif params.transform_type == conf.TransformType.DENSE:
    params.rbf_type = make_st_widget(conf.RBF_TYPE_DESCRIPTOR, label="RBF type", value=config.rbf_type)
    params.smoothness = make_st_widget(conf.SMOOTHNESS_DESCRIPTOR,
                                       label="warp field smoothness",
                                       value=config.smoothness)
    params.num_dct_coeffs = make_st_widget(conf.NUM_DCT_COEFFS_DESCRIPTOR,
                                           label="spectral coefficients per axis",
                                           value=config.num_dct_coeffs)

params.num_iterations = make_st_widget(conf.NUM_ITERATIONS_DESCRIPTOR,
                                       label="number of iterations",
                                       value=config.num_iterations)

# ++++++++++++++
#   MAIN PAGE
# ++++++++++++++

'''
# Gradient-based registration
'''


@cache_allow_output_mutation
def get_feature_map():
    return load_and_preprocess_feature_map(conf.FEATURE_MAP_DIR / params.feature_map_path, params.downscale_factor)


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


@cache_allow_output_mutation(show_spinner=False, persist=True)
def get_patch_pairs():
    return find_promising_patch_pairs(feature_map, patch_size=conf.PATCH_SIZE,
                                      stride=64 // params.downscale_factor,
                                      padding=conf.PADDING_SIZE,
                                      num_pairs=conf.NUM_PATCH_PAIRS)


patch_pairs = get_patch_pairs()
# FIXME this is hacked, it should be that the gui only sets the origins
patch_slice, window_slice, _ = patch_pairs[conf.NUM_PATCH_PAIRS - params.patch_position]
params.moving_slices = patch_slice
params.static_slices = window_slice


@cache_allow_output_mutation
def get_moving_and_static():
    return get_padded_moving_and_static(feature_map, params.moving_slices, params.static_slices)


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

if params.centroid_method == conf.CentroidMethod.EQUIDISTANT:
    '''
    Here we simply choose $n$ equidistant directions and intervals.
    '''

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


@cache_allow_output_mutation
def get_centroids_intervals():
    if params.centroid_method == conf.CentroidMethod.EQUIDISTANT:
        return get_n_equidistant_angles_and_intervals(params.num_centroids)
    elif params.centroid_method == conf.CentroidMethod.HISTOGRAM_CLUSTERING:
        return get_main_gradient_angles_and_intervals(IMAGE_FOR_MAIN_DIRECTIONS, params.kde_rho)


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
    if params.centroid_method == conf.CentroidMethod.HISTOGRAM_CLUSTERING:
        sample_points, scores = get_kde_plot_data()
    else:
        sample_points = scores = None

    plot_centroids_intervals_polar(centroids, intervals, centroids_degrees,
                                   centroids_colors, sample_points, scores)
    return figure_to_image()


''
st.image(image=plot_angles())

'''
### Binary assignments
'''

if params.filter_method == conf.FilterMethod.FARID_DERIVATIVE:
    get_binary_assignments = get_binary_assignments_from_centroids
    get_memberships = get_memberships_from_centroids
elif params.filter_method == conf.FilterMethod.GABOR:
    get_binary_assignments = partial(get_binary_assignments_from_gabor, sigma=params.gabor_filter_sigma)
    get_memberships = partial(get_memberships_from_gabor, sigma=params.gabor_filter_sigma)

centroids_degrees_and_all = ('-- all --', *centroids_degrees)
picked_angle = st.selectbox(label='angle for assignments', options=centroids_degrees_and_all)
picked_angle_index = centroids_degrees_and_all.index(picked_angle) - 1


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
    return assignments, distances, directions


static_assignments, static_distances, static_directions = get_static_assignments_distances_directions()

write_centroid_legend()


@cache_allow_output_mutation
def plot_assignments():
    plot_multiple_binary_assignments((moving_assignments, static_assignments), centroids,
                                     picked_angle_index, ('moving', 'static'))
    return figure_to_image()


st.image(image=plot_assignments(), use_column_width=True)

if params.assignment_type == conf.AssignmentType.MEMBERSHIPS:
    r'''
    ### Memberships
    Instead of using binary assignments in the moving image, we will use soft-assignments for each
    (pixel, angle) pair. This gives the following memberships for each pixel of the moving image:
    '''

    picked_angle = st.selectbox(label='angle', options=centroids_degrees_and_all)
    picked_angle_index = centroids_degrees_and_all.index(picked_angle) - 1

    write_centroid_legend()


    @cache_allow_output_mutation
    def plot_memberships_fn():
        plot_memberships(moving, moving_memberships, centroids_colors, picked_angle_index)
        return figure_to_image()


    st.image(image=plot_memberships_fn())

'''
### Fit a transformation
'''

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


# @cache_allow_output_mutation
def plot_binary_correspondences():
    plt.figure()
    if params.assignment_type == conf.AssignmentType.BINARY:
        memberships = moving_assignments
        name = "binary assignments"
    elif params.assignment_type == conf.AssignmentType.MEMBERSHIPS:
        memberships = moving_memberships
        name = "memberships"

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
    plt.title(f"correspondences from {name}")
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

transform_dof = None
if params.transform_type == conf.TransformType.LINEAR:
    transform_dof = 8
elif params.transform_type == conf.TransformType.DENSE:
    if params.num_dct_coeffs is None:
        transform_dof = 2 * moving.size
    else:
        transform_dof = 2 * (params.num_dct_coeffs ** 2)

# st.sidebar.markdown(f"The resulting transform has {transform_dof} degrees of freedom.")


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

    "### Final blending result"

    def get_blend_plot():
        _, axs = plt.subplots(1, 3, figsize=(9, 4))
        texture_path = texture_path_dict[config.feature_map_path]
        texture = imageio.imread(texture_path)[..., :3]
        texture = skimage.transform.downscale_local_mean(texture, (config.downscale_factor,
                                                                   config.downscale_factor, 1)).astype(np.uint8)

        transform = run_result.results[-1].stacked_transform
        padded_patch_slice = pad_slices(patch_slice, conf.PADDING_SIZE)
        moving_texture = texture[padded_patch_slice].astype(np.float64)

        mask = apply_transform(np.ones(moving_texture.shape[:2]), transform).astype(np.uint8)
        mask = mask[conf.PADDING_SIZE:-conf.PADDING_SIZE,
               conf.PADDING_SIZE:-conf.PADDING_SIZE]

        for c in range(texture.shape[2]):
            moving_texture[..., c] = apply_transform(moving_texture[..., c],
                                                     transform,
                                                     mode='reflect',
                                                     order=2)

        moving_texture = moving_texture[conf.PADDING_SIZE:-conf.PADDING_SIZE,
                         conf.PADDING_SIZE:-conf.PADDING_SIZE].astype(np.uint8)
        static_texture = texture[window_slice]
        lamb = 0.5
        # lamb = np.linspace(1,0,num=conf.PATCH_SIZE)[None,:,None]
        blended_texture = histogram_preserved_blending(moving_texture,
                                                       static_texture,
                                                       lamb)
        # blended_texture = np.max(np.array([moving_texture, static_texture]), axis=0)
        # blended_texture = (0.5 * moving_texture + 0.5 * static_texture).astype(np.uint8)

        axs[0].imshow(moving_texture * mask[..., None], vmin=0, vmax=255)
        axs[1].imshow(blended_texture * mask[..., None], vmin=0, vmax=255)
        axs[2].imshow(static_texture * mask[..., None], vmin=0, vmax=255)
        axs[0].set_title("warped source")
        axs[1].set_title("blend")
        axs[2].set_title("target")
        plt.tight_layout()
        return figure_to_image()

    st.image(get_blend_plot(), use_column_width=True)


if config in configs:
    config = configs[configs.index(config)]
    st.sidebar.text("Calculation done.")
    if config.is_favorite:
        st.sidebar.markdown("*Config is saved as favorite.*")
    elif st.sidebar.button("Save as favorite"):
        config = RunConfiguration(**{**attr.asdict(config), 'is_favorite': True})
        config.save()

    load_config_and_show()

else:
    button_slot = st.sidebar.empty()
    if button_slot.button("Run calculation"):
        pbar = StreamlitProgressWrapper(total=config.num_iterations)
        results = run_config(config, pbar)

        if results is None:
            st.error("Failed to run config!")
            button_slot.text("Run failed.")
        else:
            configs.append(config)
            button_slot.text("Calculation done.")
            load_config_and_show()
    else:
        st.info("Click 'Run calculation' in the sidebar to get results.")
