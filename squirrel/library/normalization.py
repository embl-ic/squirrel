
import numpy as np


def _validate_pair(value, name, *, unit=False):
    """Validate two strictly increasing finite real values."""
    if not isinstance(value, (tuple, list, np.ndarray)):
        raise TypeError(f'{name} must be a pair of real numbers')
    values = np.asarray(value)
    if values.shape != (2,) or values.dtype.kind not in 'iuf':
        raise TypeError(f'{name} must be a pair of real numbers')
    low, high = map(float, values)
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError(f'{name} must be finite and strictly increasing')
    if unit and (low < 0 or high > 1):
        raise ValueError(f'{name} must lie in [0, 1]')
    return low, high


def get_normalization_limits(image, method='minmax', quantiles=(0.01, 0.99), mask=None, ignore_zeros=False):
    """Return finite intensity limits; return None for empty/constant selections."""
    if method not in ('minmax', 'quantile'):
        raise ValueError(f'Unknown normalization method: {method!r}')
    quantiles = _validate_pair(quantiles, 'quantiles', unit=True)
    if not isinstance(ignore_zeros, (bool, np.bool_)):
        raise TypeError('ignore_zeros must be boolean')
    data = np.asarray(image)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != data.shape or mask.dtype.kind != 'b':
            raise ValueError('mask must be boolean and match image shape')
        data = data[mask]
    data = data[np.isfinite(data)]
    if ignore_zeros:
        data = data[data != 0]
    if data.size == 0:
        return None
    low, high = (np.quantile(data, quantiles) if method == 'quantile'
                 else (np.min(data), np.max(data)))
    if low >= high:
        return None
    return float(low), float(high)


def get_histogram_peak(image, smoothing_sigma=3.0, search_range=None,
                       bins=256, mask=None, ignore_zeros=False):
    """Estimate dominant histogram mode and full width at half maximum.

    Return None if the selected data or peak width is degenerate.
    """
    from scipy.ndimage import gaussian_filter1d

    if not np.isfinite(smoothing_sigma) or smoothing_sigma < 0:
        raise ValueError('peak_smoothing_sigma must be finite and nonnegative')
    if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 3:
        raise ValueError('peak_bins must be an integer >= 3')
    if search_range is not None:
        search_range = _validate_pair(search_range, 'peak_search_range')

    data = np.asarray(image)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != data.shape or mask.dtype.kind != 'b':
            raise ValueError('mask must be boolean and match image shape')
        data = data[mask]
    data = data[np.isfinite(data)]
    if ignore_zeros:
        data = data[data != 0]
    if data.size == 0 or np.min(data) == np.max(data):
        return None

    hist, edges = np.histogram(data, bins=bins)
    hist = gaussian_filter1d(hist.astype(np.float64), smoothing_sigma)
    centers = (edges[:-1] + edges[1:]) / 2
    eligible = np.ones(centers.shape, dtype=bool)
    if search_range is not None:
        eligible = (centers >= search_range[0]) & (centers <= search_range[1])
    if not np.any(eligible):
        return None
    peak_index = np.argmax(np.where(eligible, hist, -np.inf))
    peak_height = hist[peak_index]
    half_height = peak_height / 2

    left = peak_index
    while left > 0 and hist[left] >= half_height:
        left -= 1
    right = peak_index
    while right < len(hist) - 1 and hist[right] >= half_height:
        right += 1
    if left == 0 and hist[left] >= half_height:
        return None
    if right == len(hist) - 1 and hist[right] >= half_height:
        return None
    if left == peak_index or right == peak_index:
        return None

    left_crossing = np.interp(half_height, hist[left:left + 2], centers[left:left + 2])
    right_crossing = np.interp(half_height, hist[right - 1:right + 1][::-1],
                               centers[right - 1:right + 1][::-1])
    width = right_crossing - left_crossing
    if not np.isfinite(width) or width <= 0:
        return None
    return float(centers[peak_index]), float(width)


def normalize_image(image, method='minmax', quantiles=(0.01, 0.99),
                    normalization_range=None, anchors=(0.0, 1.0),
                    output_dtype='float32', mask=None, ignore_zeros=False,
                    keep_zeros=False, peak_target=0.5,
                    peak_width_target=0.15, peak_smoothing_sigma=3.0,
                    peak_search_range=None):
    """Linearly map input limits to fractional output anchors, clipping to dtype range.

    Integer outputs use their full dtype range; floating outputs use [0, 1].
    Degenerate or empty estimation sets produce zeros. Mask only selects pixels
    for limit estimation; keep_zeros restores original zero pixels.
    """
    data = np.asarray(image)
    dtype = np.dtype(output_dtype)
    if dtype.kind not in 'fiu':
        raise TypeError('output_dtype must be a real numeric dtype')
    anchors = _validate_pair(anchors, 'anchors', unit=True)
    if method not in ('minmax', 'quantile', 'peak'):
        raise ValueError(f'Unknown normalization method: {method!r}')
    _validate_pair(quantiles, 'quantiles', unit=True)
    if not isinstance(keep_zeros, (bool, np.bool_)):
        raise TypeError('keep_zeros must be boolean')
    if method == 'peak':
        if normalization_range is not None:
            raise ValueError('normalization_range is not supported for peak normalization')
        if anchors != (0.0, 1.0):
            raise ValueError('anchors are not supported for peak normalization')
        if not np.isfinite(peak_target) or not 0 <= peak_target <= 1:
            raise ValueError('peak_target must lie in [0, 1]')
        if not np.isfinite(peak_width_target) or peak_width_target <= 0:
            raise ValueError('peak_width_target must be positive and finite')
        peak = get_histogram_peak(data, smoothing_sigma=peak_smoothing_sigma,
                                  search_range=peak_search_range, mask=mask,
                                  ignore_zeros=ignore_zeros)
        if peak is None:
            return np.zeros(data.shape, dtype=dtype)
        center, width = peak
        fraction = (data.astype(np.float64) - center) * (peak_width_target / width) + peak_target
        fraction = np.clip(np.nan_to_num(fraction, nan=0., posinf=1., neginf=0.), 0, 1)
        if dtype.kind in 'iu':
            info = np.iinfo(dtype)
            fraction = fraction * (float(info.max) - float(info.min)) + float(info.min)
        result = fraction.astype(dtype)
        if keep_zeros:
            result[data == 0] = 0
        return result
    limits = (_validate_pair(normalization_range, 'normalization_range')
              if normalization_range is not None else
              get_normalization_limits(data, method, quantiles, mask, ignore_zeros))
    if limits is None:
        return np.zeros(data.shape, dtype=dtype)
    low, high = limits
    work = data.astype(np.float64)
    fraction = (work - low) / (high - low)
    fraction = anchors[0] + fraction * (anchors[1] - anchors[0])
    fraction = np.clip(np.nan_to_num(fraction, nan=0., posinf=1., neginf=0.), 0, 1)
    if dtype.kind in 'iu':
        info = np.iinfo(dtype)
        fraction = fraction * (float(info.max) - float(info.min)) + float(info.min)
    result = fraction.astype(dtype)
    if keep_zeros:
        result[data == 0] = 0
    return result


def _get_quantiles(array, quantiles=(0.1, 0.9), threshold=(1, 254), dilate_background=0):
    from copy import deepcopy
    this_array = deepcopy(array)
    if dilate_background:
        from vigra.filters import discErosion, gaussianSmoothing
        mask = 1 - discErosion((gaussianSmoothing(this_array, 1.0) > 1).astype('uint8'), dilate_background)
        this_array[mask > 0] = 0
    array_ = this_array[np.logical_and(this_array > threshold[0], this_array < threshold[1])]
    return np.quantile(array_, quantiles[0]), np.quantile(array_, quantiles[1])


def apply_quantiles(image, quantiles=(0.1, 0.9), anchors=(0.2, 0.8), dilate_background=0, keep_zeros=False):
    """Normalize quantiles while excluding saturated/background pixels."""
    image = np.asarray(image)
    if image.dtype.kind not in 'iu':
        raise TypeError('apply_quantiles requires an integer image')
    max_value = np.iinfo(image.dtype).max
    estimation_mask = (image > 1) & (image < max_value - 1)
    if dilate_background:
        from vigra.filters import discErosion, gaussianSmoothing
        foreground = (gaussianSmoothing(image, 1.0) > 1).astype('uint8')
        background = 1 - discErosion(foreground, dilate_background)
        estimation_mask &= background == 0
    return normalize_image(image, method='quantile', quantiles=quantiles,
                           anchors=anchors, mask=estimation_mask,
                           keep_zeros=keep_zeros, output_dtype='uint8')


def _process_stack_slices(stack, image_function, image_args=(), z_range=None, n_workers=1, opencv=False):
    """Apply an image function to selected slices in serial or process workers."""
    from squirrel.library.data import norm_z_range

    start, stop = norm_z_range(z_range, stack.shape[0])
    if not isinstance(n_workers, int) or n_workers < 1:
        raise ValueError('n_workers must be a positive integer')
    if n_workers == 1:
        return np.asarray([image_function(stack[idx], *image_args) for idx in range(start, stop)])

    from multiprocessing import Pool, get_context

    print(f'Running with {n_workers} CPUs')
    if opencv:
        pool_factory = get_context('spawn').Pool
        pool_kwargs = {'initializer': _init_opencv_worker}
    else:
        pool_factory = Pool
        pool_kwargs = {}

    with pool_factory(processes=n_workers, **pool_kwargs) as pool:
        tasks = (pool.apply_async(image_function, (stack[idx], *image_args)) for idx in range(start, stop))
        return np.asarray([task.get() for task in tasks])


def _init_opencv_worker():
    """Prevent nested OpenCV threading in spawned CLAHE workers."""
    import cv2
    cv2.setNumThreads(1)


def normalize_slices(
        stack,
        dilate_background=0,
        quantiles=(0.1, 0.9),
        anchors=(0.2, 0.8),
        keep_zeros=False,
        z_range=None,
        n_workers=1
):
    # FIXME implement for other data types
    # assert stack.dtype == 'uint8', f'Normalization only implemented for uint8 data, found {stack.dtype}'
    if stack.dtype != 'uint8':
        print('Warning: Normalization will convert the dtype to uint8! ')
    return _process_stack_slices(stack, apply_quantiles,
                                 (quantiles, anchors, dilate_background, keep_zeros), z_range, n_workers)


def _adjust_greyscale_in_image(img, greys_in=None, greys_out=None, cast_dtype='uint8'):
    """Map an input grey interval to an output grey interval."""
    dtype = np.dtype(cast_dtype)
    if dtype.kind not in 'iu':
        raise TypeError('adjust_greyscale requires an integer output dtype')
    info = np.iinfo(dtype)
    if greys_out is None:
        greys_out = (info.min, info.max)
    if greys_in is None:
        greys_in = (np.min(img), np.max(img))
    if greys_out[0] == greys_out[1]:
        raise ValueError('greys_out must contain distinct values')
    if greys_out[0] < info.min or greys_out[1] > info.max:
        raise ValueError('greys_out must be within the output dtype range')
    anchors = ((greys_out[0] - info.min) / (info.max - info.min),
               (greys_out[1] - info.min) / (info.max - info.min))
    return normalize_image(img, normalization_range=greys_in, anchors=anchors,
                           output_dtype=dtype)


def adjust_greyscale(
        stack,
        z_range=None,
        greys_in=None,
        greys_out=None,
        cast_dtype='uint8',
        n_workers=1
):

    return _process_stack_slices(stack, _adjust_greyscale_in_image,
                                 (greys_in, greys_out, cast_dtype), z_range, n_workers)


def clahe_on_image(
        image,
        clip_limit=3.0,
        tile_grid_size=(127, 127),
        cast_dtype=None,
        invert_output=False,
        gaussian_sigma=0.0,
        auto_mask=False,
        background_to_mean=False,
        tile_grid_in_pixels=False
):
    image = np.asarray(image)
    if image.ndim != 2 or image.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
        raise ValueError(
            'CLAHE requires a 2D uint8 or uint16 image; '
            f'got shape {image.shape} and dtype {image.dtype}. '
            'Normalize and cast the image before CLAHE, for example with '
            'ImageFilter.same(image, normalize=True, '
            'normalization_method="quantile", cast_dtype="uint8").'
        )

    if tile_grid_in_pixels:
        tile_grid_size = (np.array(image.shape) / np.array(tile_grid_size)).astype(int).tolist()

    from cv2 import createCLAHE
    clahe = createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size) 

    mask = None
    if auto_mask:
        mask = image > 0
    if background_to_mean:
        tmp_image = image.copy()
        tmp_image[~mask] = np.mean(image[mask])
    else:
        # Only make a shallow copy in this case
        tmp_image = image

    clahe_filtered = clahe.apply(tmp_image)
    dtype_in = clahe_filtered.dtype

    if auto_mask:
        clahe_filtered[~mask] = 0

    if gaussian_sigma > 0.0:
        from vigra.filters import gaussianSmoothing
        clahe_filtered = gaussianSmoothing(clahe_filtered.astype('float32'), gaussian_sigma).astype(dtype_in)
    if invert_output:
        from squirrel.library.volume import invert_image
        clahe_filtered = invert_image(clahe_filtered)

    if cast_dtype is None:
        return clahe_filtered

    dtype_out = np.dtype(cast_dtype)
    max_val_in = np.iinfo(dtype_in).max
    max_val_out = np.iinfo(dtype_out).max
    return (clahe_filtered.astype('float64') / max_val_in * max_val_out).astype(dtype_out)


def clahe_on_slices(
        stack,
        clip_limit=3.0,
        tile_grid_size=(127, 127),
        cast_dtype=None,
        invert_output=False,
        gaussian_sigma=0.0,
        auto_mask=False,
        background_to_mean=False,
        tile_grid_in_pixels=False,
        z_range=None,
        n_workers=1
):

    image_args = (clip_limit, tile_grid_size, cast_dtype, invert_output, gaussian_sigma,
                  auto_mask, background_to_mean, tile_grid_in_pixels)
    return _process_stack_slices(stack, clahe_on_image, image_args, z_range, n_workers, opencv=True)


if __name__ == '__main__':
    # from squirrel.library.io import load_data_handle
    # h, _ = load_data_handle('/media/julian/Data/tmp/tmp_hela_16bit_3slices/')
    # normalize_slices(
    #         h,
    #         dilate_background=0,
    #         quantiles=(0.1, 0.9),
    #         anchors=(0.2, 0.8),
    #         z_range=None,
    #         n_workers=1
    # )

    from squirrel.library.io import read_tif_slice
    img = read_tif_slice(filepath='/mnt/icem/hennies/tmp/clahe_test/image.tif', return_filepath=False)

    from matplotlib import pyplot as plt
    plt.imshow(img, cmap='gray')

    clahe = clahe_on_image(
        img,
        cast_dtype='uint8',
        gaussian_sigma=2,
        auto_mask=True,
        tile_grid_size=(63, 63),
        tile_grid_in_pixels=True
    )
    plt.figure()
    plt.imshow(clahe, cmap='gray')
    plt.show()
