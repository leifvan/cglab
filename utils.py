from matplotlib import pyplot as plt
import random
import string
import imageio
import os
import math


def plot_diff(warped, target):
    """
    Creates a figure with 3 images (from left to right):

    - the warped image
    - the difference between warped and target, and
    - the target image.

    Note: ``plt.show()`` will not be called by this function.

    :param warped: The warped image.
    :param target: The target image.
    """
    _, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(warped, cmap='coolwarm', vmin=-1, vmax=1)
    axs[1].imshow(warped - target, cmap='coolwarm')
    axs[2].imshow(-target, cmap='coolwarm', vmin=-1, vmax=1)

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()


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

    def add_current_fig(self):
        """
        Add the current matplotlib figure as the next image in the gif.
        :return:
        """
        random_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        path = f"data/temp/{random_filename}.png"
        plt.savefig(path)
        self.image_paths.append(path)

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
    def __init__(self, n, key, reverse=False):
        self.cur_min = math.inf
        # self.cur_max = -math.inf
        self.items = []
        self.n = n
        self.key = key
        self.key_factor = -1 if reverse else 1

    def _key(self, value):
        return self.key_factor * self.key(value)

    def add(self, item):
        if len(self.items) < self.n:
            self.items.append(item)
            self.items.sort(key=self._key)
        elif self._key(item) > self._key(self.items[0]):
            self.items.pop(0)
            self.items.append(item)
            self.items.sort(key=self._key)
