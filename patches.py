from itertools import combinations
from math import ceil
from operator import itemgetter

import numpy as np
from tqdm import tqdm

from utils import NBestCollection

rng = np.random.default_rng()


def patches_iterator(height, width, patch_size, stride=1, padding=0):
    for y in range(padding, height - patch_size - padding, stride):
        for x in range(padding, width - patch_size - padding, stride):
            yield tuple([slice(y, y + patch_size), slice(x, x + patch_size)])


def find_promising_patch_pairs(feature_map, patch_size, stride=1, padding=0, num_pairs=10):
    num_patches = ceil((feature_map.shape[0] - patch_size) / stride) * ceil(
        (feature_map.shape[1] - patch_size) / stride)
    num_patch_pairs = num_patches * (num_patches - 1) // 2

    best_indices = NBestCollection(n=num_pairs, key=itemgetter(2), reverse=True)

    pairs_iterator = combinations(patches_iterator(*feature_map.shape[:2], patch_size, stride, padding), r=2)
    for i, (slice_a, slice_b) in enumerate(tqdm(pairs_iterator, total=num_patch_pairs)):
        dist = np.linalg.norm(feature_map[slice_a] - feature_map[slice_b])
        best_indices.add((slice_a, slice_b, dist))

    return tuple(best_indices.items)


def generate_random_tiling_indices(output_shape, input_shape, patch_shape, overlap):
    out_height, out_width = output_shape
    in_height, in_width = input_shape
    p_height, p_width = patch_shape
    overlap_y, overlap_x = overlap

    num_patches_v = ceil(out_height / (p_height - overlap_y))
    num_patches_h = ceil(out_width / (p_width - overlap_x))
    num_patches = num_patches_v * num_patches_h

    origins = rng.integers(0, (in_height-p_height, in_width-p_width), (num_patches, 2))
    return origins.reshape((num_patches_v, num_patches_h, 2))


