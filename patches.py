import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from scipy.spatial.distance import pdist
from itertools import combinations
from tqdm import tqdm
from math import ceil
from utils import NBestCollection
from operator import itemgetter


def get_num_patches(shape, patch_size, stride):
    return sum(1 for _ in range(0, shape[0]-patch_size, stride)
                 for _ in range(0, shape[1]-patch_size, stride))


def patches_iterator(feature_map, patch_size, stride=1):
    # TODO make this a coordinate iterator and write an extra function for extraction
    height, width = feature_map.shape[:2]
    for y in range(0, height-patch_size, stride):
        for x in range(0, width-patch_size, stride):
            yield feature_map[y:y+patch_size, x:x+patch_size]


def find_promising_patch_pairs(feature_map, patch_size, stride=1):
    print(feature_map.shape, patch_size, stride)
    num_patches = ceil((feature_map.shape[0]-patch_size) / stride) * ceil((feature_map.shape[1]-patch_size) / stride)
    num_patch_pairs = num_patches * (num_patches - 1) // 2

    # TODO if we only take 10 best we don't need to store all
    distances = np.zeros(num_patch_pairs, dtype=np.float32)
    best_indices = NBestCollection(n=10, key=itemgetter(1))

    pairs_iterator = combinations(patches_iterator(feature_map, patch_size, stride), r=2)
    for i, (patch_a, patch_b) in enumerate(tqdm(pairs_iterator, total=num_patch_pairs)):
        dist = np.linalg.norm(patch_a - patch_b)
        best_indices.add((i, dist))

    # all_patches = extract_patches_2d(feature_map, (patch_size, patch_size))  # (n_patches, h, w)
    # all_patches = np.reshape(all_patches, (all_patches.shape[0], -1))
    # distances = pdist(all_patches, metric='euclidean')  # distances[i*j] = distance of patch i and j
    best_indices = np.array(best_indices.items, np.int)
    return np.unravel_index(best_indices[:,0], shape=(num_patches, num_patches))