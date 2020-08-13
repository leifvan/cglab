"""
Code for histogram-preserved blending as described by Deliot and Heitz in "Procedural stochastic
textures by tiling and blending." (2019).
https://eheitzresearch.wordpress.com/738-2/
"""

import numpy as np
import scipy.stats as sp_stats


def _cdf(x):
    return sp_stats.norm.cdf(x, loc=0.5, scale=1 / 6)


def _icdf(u):
    return sp_stats.norm.ppf(u, loc=0.5, scale=1 / 6)


def get_decorrelation_params(image):
    """
    Returns parameters for a decorrelation of the colors in ``image``.
    :param image: An array of shape (height, width, num_channels) with integer color values in [0, 255].
    :return: Parameter tuple for the :meth:`decorrelate_colors` method.
    """
    # TODO this can surely be improved
    n_channels = image.shape[2]
    pixels = np.reshape(image, (-1, n_channels)).copy().astype(np.float) / 255
    mean = pixels.mean(axis=0)
    pixels -= mean
    cov = np.cov(pixels.T)
    w, v = np.linalg.eigh(cov)
    mat = v @ np.diag(w ** (-0.5))
    pixels_transformed = pixels @ mat
    pixel_min = pixels_transformed.min(axis=0)
    pixel_ptp = pixels_transformed.max(axis=0) - pixel_min
    origin = pixel_min * (1 / pixel_ptp)
    return mat @ np.diag(1 / pixel_ptp), origin, mean


def decorrelate_colors(image, mat, origin, mean):
    """
    Decorrelates the colors in the given ``image`` based on the parameters.

    :param image: The input image of shape (height, width, num_channels) with integer color values
        in [0, 255].
    :param mat: A (3,3) transformation matrix that aligns the principal components of the color
        space with the axes and normalizes the values to [-0.5, 0.5] along these.
    :param origin: A transposition vector to be applied after multiplication with ``mat``.
    :param mean: A transposition vector to be applied before multiplication with ``mat``.
    :return: The decorrelated image of shape (height, width, num_channels) with integer color
        values in [0, 255].
    """
    n_channels = image.shape[2]
    pixels = np.reshape(image, (-1, n_channels)).copy().astype(np.float) / 255
    decorrelated = (pixels - mean) @ mat - origin
    return np.reshape(decorrelated * 255, image.shape).astype(np.int)


def recorrelate_colors(image, mat, origin, mean):
    """
    Reverses the decorrelation of ``image`` with the given parameters.

    :param image: The input image of shape (height, width, num_channels) with integer color values
        in [0, 255].
    :param mat: A (num_channels, num_channels) transformation matrix that aligns the principal
        components of the color space with the axes and normalizes the values to [-0.5, 0.5]
        along these.
    :param origin: A transposition vector to be applied after multiplication with ``mat``.
    :param mean: A transposition vector to be applied before multiplication with ``mat``.
    :return: The recorrelated image of shape (height, width, num_channels) with integer color
        values in [0, 255].
    """
    recorrelated = (image.astype(np.float) / 255 + origin) @ np.linalg.inv(mat) + mean
    recorrelated = (recorrelated * 255).astype(np.int)
    return recorrelated


def get_degauss_lut(image):
    """
    Computes a lookup table used to reserve the Gaussian histogram transformation of the given
    ``image``.

    :param image: The input image of shape (height, width, num_channels) with integer color values
        in [0, 255].
    :return: The lookup table of shape (256, num_channels).
    """
    n_channels = image.shape[2]
    assert n_channels in (1, 3)
    n_pixels = image.shape[0] * image.shape[1]

    lut = np.zeros((256, n_channels))
    g_lut = (np.arange(256) + 0.5) / 256
    u_lut = _cdf(g_lut)

    for c in range(n_channels):
        sorted_indices = np.argsort(image[..., c], axis=None)
        ii, jj = np.unravel_index(sorted_indices, image.shape[:2])
        lut_indices = np.floor(u_lut * n_pixels).astype(np.int)
        lut[:, c] = image[ii, jj, c][lut_indices]

    return lut


def gaussify_colors(image):
    """
    Applies a histogram transformation on each color channel of ``image`` s.t. the histogram
    of the channel equals a Gaussian distribution with mu = 0.5 and sigma = 1/6.

    :param image: The input image of shape (height, width, num_channels) with integer color values
        in [0, 255].
    :return: The gaussified image of shape (height, width, num_channels) with integer color values
        in [0, 255].
    """
    n_channels = image.shape[2]
    assert n_channels in (1, 3)
    n_pixels = image.shape[0] * image.shape[1]
    image_gaussified = np.zeros_like(image, dtype=np.int)

    u_img = (np.arange(n_pixels, dtype=np.float) + 0.5) / n_pixels
    g_img = _icdf(u_img)

    for c in range(n_channels):
        sorted_indices = np.argsort(image[..., c], axis=None)
        ii, jj = np.unravel_index(sorted_indices, image.shape[:2])
        image_gaussified[ii, jj, c] = g_img * 255

    return image_gaussified.clip(0, 255)


def degaussify_colors(image, lut):
    """
    Reverses the Gaussian histogram transformation in ``image`` using ``lut``.

    :param image: The gaussified image of shape (height, width, num_channels) with integer color
        values in [0, 255].
    :param lut: The lookup table of shape (256, num_channels).
    :return: The degaussified image of shape (height, width, num_channels) with integer color values
        in [0, 255].
    """
    n_channels = image.shape[2]
    assert n_channels in (1, 3)

    image = np.zeros_like(image)
    for c in range(n_channels):
        image[..., c] = lut[image[..., c], c]

    return image


def histogram_preserved_blending(patch1, patch2, lamb):
    """
    Blends two patches of the same image using histogram-preserved blending with decorrelated
    color spaces. The correlation information and lookup table for the histogram transform is
    taken from ``patch1``.

    The patches are blended blended like ``lamb * patch1 + (1 - lamb) * patch2`` in the decorrelated
    and gaussified color space and transformed back into the original color space afterwards.

    :param patch1: An of shape (height, width, num_channels) with integer color
        values in [0, 255].
    :param patch2: Another image of shape (height, width, num_channels) with integer color
        values in [0, 255].
    :param lamb: The blending factor(s). Can be a single value or an broadcastable to the patches.
    :return: The blended image of shape (height, width, num_channels) with integer color
        values in [0, 255].
    """
    assert np.all((lamb >= 0) & (lamb <= 1))

    d_params = get_decorrelation_params(patch1)

    patches = [patch1, patch2]
    decorrelated = [decorrelate_colors(p, *d_params) for p in patches]

    lut = get_degauss_lut(decorrelated[0])
    gaussified = [gaussify_colors(d) for d in decorrelated]
    blended = gaussified[0] * lamb + gaussified[1] * (1 - lamb)
    degaussified = degaussify_colors(blended.astype(np.int), lut)
    recorrelated = recorrelate_colors(degaussified, *d_params)
    return recorrelated


if __name__ == '__main__':
    import imageio
    import matplotlib.pyplot as plt

    image = imageio.imread("data/cobblestone_floor_03_diff_1k.png")[..., :3]
    plt.imshow(image)
    plt.show()

    d_params = get_decorrelation_params(image)
    decorrelated = decorrelate_colors(image, *d_params)
    plt.imshow(decorrelated)
    plt.show()

    gaussified = gaussify_colors(decorrelated)
    plt.imshow(gaussified, vmin=0, vmax=255)
    plt.show()

    lut = get_degauss_lut(decorrelated)
    degaussified = degaussify_colors(gaussified, lut)
    plt.imshow(degaussified)
    plt.show()

    recorrelated = recorrelate_colors(degaussified, *d_params)
    plt.imshow(recorrelated)
    plt.show()

    print(np.linalg.norm(image - recorrelated))  # result is < 1580

    patch1 = image[300:500, 100:300]
    patch2 = image[700:900, 300:500]
    blend_linear = (0.5 * patch1 + 0.5 * patch2).astype(np.int)
    blend_hp = histogram_preserved_blending(patch1, patch2, 0.5)

    plt.imshow(blend_linear)
    plt.show()
    plt.imshow(blend_hp)
    plt.show()