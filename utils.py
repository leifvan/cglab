import math
import os
import random
import string

import attr
import imageio
import matplotlib.cm as plt_cm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import CSS4_COLORS, to_rgb

CSS_INDIANRED_RGB = to_rgb(CSS4_COLORS['indianred'])
CSS_STEELBLUE_RGB = to_rgb(CSS4_COLORS['steelblue'])
CSS_GOLD_RGB = to_rgb(CSS4_COLORS['gold'])


def pad_slices(slices, padding, assert_shape=None):
    """
    Pads a list of slices with the ``padding`` value, e.g. if ``slices=[[5:8],[2:4]]`` and
    ``padding = 2``, it will return slices ``[[3:10, [0:6]]``. If ``assert_shape`` is not ``None``,
    it will check if left bounds of the padded slices are non-negative and the i-th padded slice
    stops before value ``assert_shape[i]``.

    :param slices: A list of ``slice`` object.
    :param padding: An integer.
    :param assert_shape: ``None`` or a list of upper bounds for the stop values of the slices.
    :return: A list of padded slices.
    """
    new_slices = tuple(slice(s.start - padding, s.stop + padding, s.step) for s in slices)
    if assert_shape:
        assert all(0 <= s.start < shape and 0 <= s.stop < shape
                   for s, shape in zip(new_slices, assert_shape))
    return new_slices


class GifExporter:
    """
    A utility class for exporting matplotlib figures as a gif.
    """

    def __init__(self):
        self.image_paths = []
        os.makedirs("data/temp/", exist_ok=True)

    def _get_random_filename(self):
        random_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        return f"data/temp/{random_filename}.png"

    def add_current_fig(self):
        """
        Add the current matplotlib figure as the next image in the gif.
        :return:
        """
        random_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        path = f"data/temp/{random_filename}.png"
        plt.savefig(path)
        self.image_paths.append(path)

    def add_image(self, image=None, path=None):
        if image is not None and path is None:
            path = self._get_random_filename()
            imageio.imwrite(path, image)
            self.image_paths.append(path)
        elif image is None and path is not None:
            # try to read to check if it is a valid image
            imageio.imread(path)
            self.image_paths.append(path)
        raise Exception("Either image or path must be None.")

    def save_gif(self, path, duration=None):
        """
        Saves the added figures as a gif at ``path``.

        :param path: The location to save the gif to.
        :param duration: An optional duration (in seconds). Every image in the gif will be shown
            for that long.
        """
        with imageio.get_writer(path, mode='I', duration=duration) as writer:
            for path in self.image_paths:
                image = imageio.imread(path)
                writer.append_data(image)

        for path in self.image_paths:
            os.remove(path)


class NBestCollection:
    """
    A helper class to keep the n highest valued items (according to a key function) of a collection
    in an online fashion.

    The kept items are stored in the instance attribute ``items``.
    """

    def __init__(self, n, key=lambda x: x, reverse=False):
        """
        :param n: Number of best items to keep.
        :param key: An optional key function that returns a value for items to be added. If not
            given, it will use the numerical value of the items.
        :param reverse: If ``True``, the negative value of ``key`` will be used, i.e. the n lowest
            valued items are kept.
        """
        self.items = []
        """List of n best items, ordered from low (``self.items[0]``) to high (``self.items[-1]``)."""
        self.n = n
        """Number of best items to keep."""
        self.key = key
        """Function that returns the value of an item."""

        self.key_factor = -1 if reverse else 1

    def _key(self, value):
        return self.key_factor * self.key(value)

    def add(self, item):
        """
        Adds the given ``item`` if its value is one of n-th highest of the items seen so far.

        :param item: The item to add.
        """
        if len(self.items) < self.n:
            self.items.append(item)
            if len(self.items) == self.n:
                self.items.sort(key=self._key)
        elif self._key(item) > self._key(self.items[0]):
            self.items.pop(0)
            self.items.append(item)
            self.items.sort(key=self._key)


def get_quadratic_subplot_for_n_axes(n, raveled_axes_only=False):
    sqrt_n = math.ceil(math.sqrt(n))
    fig, axs = plt.subplots(sqrt_n, sqrt_n, figsize=(3 * sqrt_n, 4 * sqrt_n))
    if raveled_axes_only:
        return axs.ravel()

    return fig, axs


def tight_layout_with_suptitle():
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def get_colored_difference_image(moving=None, static=None):
    # TODO use the moving, static names everywhere in the code
    if moving is None:
        moving = np.zeros_like(static)
    elif static is None:
        static = np.zeros_like(moving)

    image = np.ones((*moving.shape[:2], 3))
    image[:] = to_rgb(CSS4_COLORS['white'])
    moving_mask = np.isclose(moving, 1)
    static_mask = np.isclose(static, 1)
    image[moving_mask & ~static_mask] = CSS_INDIANRED_RGB
    image[~moving_mask & static_mask] = CSS_STEELBLUE_RGB
    image[moving_mask & static_mask] = CSS_GOLD_RGB
    return image


def get_slice_intersection(slices_a, slices_b):
    return tuple([slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(slices_a, slices_b)])


def angle_to_rgb(angles, with_alpha=False):
    if with_alpha:
        return plt_cm.get_cmap('hsv')((angles + np.pi) / 2 / np.pi)
    return plt_cm.get_cmap('hsv')((angles + np.pi) / 2 / np.pi)[..., :3]


@attr.s
class TransformResult:
    stacked_transform = attr.ib()
    error = attr.ib()
    energy = attr.ib()


def get_boundary_mask(shape, start, stop):
    mask = np.zeros(shape)
    for k in range(start, stop+1):
        mask[k:-k-1, k:-k-1] = (k - start) / (stop - start)
    return mask