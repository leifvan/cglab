import scipy


def dense_displacement_to_dct(displacement, k):
    return scipy.fft.dctn(displacement, axes=(1,2))[:, :k, :k]


def dct_to_dense_displacement(dct, shape):
    return scipy.fft.idctn(dct, s=shape[1:], axes=(1,2))
