"""Shared dtype helpers for filtering and other image operations."""

import numpy as np


def working_dtype(*arrays):
    """Select a float dtype suitable for arithmetic on input arrays."""
    dtypes = [np.asarray(a).dtype for a in arrays]
    if any(d.kind == 'f' and d.itemsize > 4 or d.kind in 'iu' and d.itemsize > 2 for d in dtypes):
        return np.float64
    return np.float32


def safe_cast(array, dtype):
    dtype = np.dtype(dtype)
    array = np.asarray(array)
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        # Integer casts of NaN and infinity are platform-dependent.
        array = np.nan_to_num(array, nan=0.0, posinf=float(limits.max), neginf=float(limits.min))
        array = np.clip(array, limits.min, limits.max)
    return array.astype(dtype)


def rescale_integer_dtype(array, target_dtype):
    """Map the full uint8/uint16 range to the target dtype range."""
    array = np.asarray(array)
    source_dtype = array.dtype
    target_dtype = np.dtype(target_dtype)
    supported = (np.dtype('uint8'), np.dtype('uint16'))
    if source_dtype not in supported or target_dtype not in supported:
        raise TypeError(
            f'rescale_dtype supports only uint8 and uint16 input/output; '
            f'got input {source_dtype} and output {target_dtype}. '
            'Convert the input to uint8 or uint16 first, or use '
            'normalize=True with cast_dtype for data-dependent normalization.')

    if source_dtype == target_dtype:
        return array.copy()

    source_max = np.iinfo(source_dtype).max
    target_max = np.iinfo(target_dtype).max
    # Integer arithmetic gives deterministic round-to-nearest for both
    # directions without float32 precision loss or per-image statistics.
    scaled = (array.astype(np.uint64) * target_max + source_max // 2) // source_max
    return scaled.astype(target_dtype)
