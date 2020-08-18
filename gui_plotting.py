import numpy as np
import matplotlib.pyplot as plt

from gradient_directions import plot_binary_assignments


def plot_centroids_intervals_polar(centroids, intervals, centroids_labels, centroids_colors,
                                   kde_points=None, kde_scores=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')

    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks(centroids)
    ax.set_xticklabels(centroids_labels)

    for angle, (ilow, ihigh), color in zip(centroids, intervals, centroids_colors):
        ax.plot([angle, angle], [0, 1], c=color)
        if ilow > ihigh:
            ihigh += 2 * np.pi
        ax.fill_between(np.linspace(ilow, ihigh, num=3), 0, 2, color=color, alpha=0.1)

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_theta_zero_location('W')
    ax.set_theta_direction(-1)

    if kde_points is not None and kde_scores is not None:
        ax.plot(kde_points, kde_scores)
        ax.set_ylim(0, kde_scores.max())

    return fig, ax


def plot_multiple_binary_assignments(assignments_list, centroids, i, labels):
    n = len(assignments_list)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
    if i == -1:
        for ax, assignments in zip(axs, assignments_list):
            plot_binary_assignments(assignments, centroids, ax)
    else:
        for ax, assignments in zip(axs, assignments_list):
            plot_binary_assignments(assignments[i, None], centroids[i, None], ax)

    for ax, label in zip(axs, labels):
        ax.set_title(label)

    return fig, axs


def plot_memberships(patch, memberships, centroids_colors, i):
    plt.figure()
    membership_image = np.tile(patch[..., None], reps=(1, 1, 3)) * 0.2
    if i == -1:
        for i, color in enumerate(centroids_colors):
            membership_image[memberships[i] != 0] = 0
            membership_image += memberships[i, ..., None] * color
    else:
        membership_image[memberships[i] != 0] = 0
        membership_image += memberships[i, ..., None] * centroids_colors[i]
    plt.imshow(membership_image)
