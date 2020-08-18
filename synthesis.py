import numpy as np
from patches import generate_random_tiling_indices


def simple_tiling_synthesis(texture, output_shape, patch_shape, overlap):
    indices = generate_random_tiling_indices(output_shape, texture.shape[:2], patch_shape, overlap)

    p_height, p_width = patch_shape
    op_height = p_height - overlap[0]
    op_width = p_width - overlap[1]

    output = np.zeros((output_shape[0] + p_height, output_shape[1] + p_width, texture.shape[2]))
    blend_map = np.zeros(output.shape[:2])

    # create a blend mask
    assert overlap[0] == overlap[1]
    mask = np.ones(patch_shape)
    for k in range(overlap[0]):
        mask[k:-(k+1),k:-(k+1)] = k / overlap[0]

    for j in range(indices.shape[0]):
        for i in range(indices.shape[1]):
            oy, ox = j * op_height, i * op_width
            iy, ix = indices[j, i]
            output[oy:oy + p_height, ox:ox + p_width] += texture[iy:iy + p_height, ix:ix + p_width] * mask[...,None]
            blend_map[oy:oy + p_height, ox:ox + p_width] += mask

    output = output[:output_shape[0],:output_shape[1]]
    blend_map = blend_map[:output_shape[0],:output_shape[1]]
    output /= blend_map[...,None]
    return output.astype(np.uint8)


if __name__ == '__main__':
    import imageio
    import matplotlib.pyplot as plt

    texture = imageio.imread("data/textures/blue_floor_tiles_01.png")[..., :3]
    plt.imshow(texture)
    plt.show()
    synthesized = simple_tiling_synthesis(texture, texture.shape[:2], (256, 256), (128,128))
    plt.imshow(synthesized)
    plt.show()