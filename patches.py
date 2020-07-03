import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from scipy.spatial.distance import pdist
from itertools import combinations
from tqdm import tqdm
from math import ceil
from utils import NBestCollection
from operator import itemgetter
from collections import namedtuple

PatchPair = namedtuple("PatchPair", "slice_a slice_b dist")


def get_num_patches(shape, patch_size, stride):
    return sum(1 for _ in range(0, shape[0] - patch_size, stride)
               for _ in range(0, shape[1] - patch_size, stride))


def patches_iterator(height, width, patch_size, stride=1):
    for y in range(0, height - patch_size, stride):
        for x in range(0, width - patch_size, stride):
            yield tuple([slice(y, y + patch_size), slice(x, x + patch_size)])


def find_promising_patch_pairs(feature_map, patch_size, stride=1, num_pairs=10):
    num_patches = ceil((feature_map.shape[0] - patch_size) / stride) * ceil(
        (feature_map.shape[1] - patch_size) / stride)
    num_patch_pairs = num_patches * (num_patches - 1) // 2

    best_indices = NBestCollection(n=num_pairs, key=itemgetter(2), reverse=True)

    pairs_iterator = combinations(patches_iterator(*feature_map.shape[:2], patch_size, stride), r=2)
    for i, (slice_a, slice_b) in enumerate(tqdm(pairs_iterator, total=num_patch_pairs)):
        dist = np.linalg.norm(feature_map[slice_a] - feature_map[slice_b])
        best_indices.add((slice_a, slice_b, dist))

    return tuple(best_indices.items)
