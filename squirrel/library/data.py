
import numpy as np


def invert_data(data):
    # FIXME add tests
    try:
        data = np.iinfo(data.dtype).max - data
    except ValueError:
        data = data.max() - data
    return data


def resolution_to_pixels(value, resolution):
    if type(value) in [list, tuple, np.array]:
        return np.array(value) / resolution
    return value / resolution


def norm_8bit(im, quantiles, ignore_zeros=False):
    """Normalize image quantiles to the uint8 range."""
    from squirrel.library.normalization import normalize_image
    return normalize_image(im, method='quantile', quantiles=quantiles,
                           ignore_zeros=ignore_zeros, output_dtype='uint8')


def get_contrast_limits(im, method='minmax'):
    """Return contrast limits, preserving the legacy unsupported-method result."""
    if method != 'minmax':
        return None
    from squirrel.library.normalization import get_normalization_limits
    limits = get_normalization_limits(im, method=method)
    return limits if limits is not None else (im.min(), im.max())


def norm_full_range(im, quantiles, anchors=None, ignore_zeros=False, mask=None, cast_8bit=False):
    """Normalize unsigned image quantiles, optionally estimating inside a mask."""
    from squirrel.library.normalization import normalize_image
    im = np.asarray(im)
    if im.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
        raise TypeError('norm_full_range requires uint8 or uint16 input')
    if anchors is None:
        anchors = quantiles
    output_dtype = 'uint8' if cast_8bit else im.dtype
    result = normalize_image(im, method='quantile', quantiles=quantiles,
                             anchors=anchors, ignore_zeros=ignore_zeros,
                             mask=mask, output_dtype=output_dtype)
    if mask is not None and not ignore_zeros:
        result = result.copy()
        result[~np.asarray(mask, dtype=bool)] = 0
    return result


def norm_z_range(z_range, len_stack):

    if z_range is None:
        z_range = [0, len_stack]
    if z_range[1] > len_stack:
        z_range[1] = len_stack
    if z_range[0] < 0:
        z_range[0] = 0

    return z_range


def get_optimal_dtype(inp):
    if type(inp) == np.array:
        max_value = inp.max()
    else:
        max_value = inp
    if max_value <= np.iinfo(np.uint8).max:
        return np.uint8
    elif max_value <= np.iinfo(np.uint16).max:
        return np.uint16
    elif max_value <= np.iinfo(np.uint32).max:
        return np.uint32
    else:
        return np.uint64  # Fallback to original dtype