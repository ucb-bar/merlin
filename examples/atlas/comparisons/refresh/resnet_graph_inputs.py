"""Generic NCHW patch extraction and tensor identities for graph exporters."""

from merlin.common.digest import sha256_bytes

import hashlib


def raw_identity(a):
    return {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha256_bytes(a.tobytes())}


def pad_matrix(a, rows, columns, np):
    padded = np.zeros((rows, columns), dtype=np.float32)
    padded[: a.shape[0], : a.shape[1]] = a
    return padded


def explicit_patches(activation, module, output_shape, np):
    """Independent NCHW im2col with channel/kernel-major K and spatial-major M."""
    batch, channels, height, width = activation.shape
    if batch != 1 or module.groups != 1 or module.padding_mode != "zeros":
        raise ValueError("Require one dense zero-padded NCHW image")
    oh, ow = output_shape[-2:]
    kh, kw = module.kernel_size
    positions = np.arange(oh * ow)
    h, w = positions // ow, positions % ow
    result = np.zeros((oh * ow, channels * kh * kw), dtype=np.float32)
    for c in range(channels):
        for r in range(kh):
            for s in range(kw):
                ih = h * module.stride[0] + r * module.dilation[0] - module.padding[0]
                iw = w * module.stride[1] + s * module.dilation[1] - module.padding[1]
                valid = (ih >= 0) & (ih < height) & (iw >= 0) & (iw < width)
                result[valid, (c * kh + r) * kw + s] = activation[0, c, ih[valid], iw[valid]]
    return result
