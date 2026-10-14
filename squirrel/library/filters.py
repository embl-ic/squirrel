"""Image filtering and JSON-compatible filter pipelines."""
import inspect
from functools import wraps
from multiprocessing import Pool

import numpy as np


def _working_dtype(*arrays):
    """Select a float dtype suitable for arithmetic on input arrays."""
    dtypes = [np.asarray(a).dtype for a in arrays]
    if any(d.kind == 'f' and d.itemsize > 4 or d.kind in 'iu' and d.itemsize > 2 for d in dtypes):
        return np.float64
    return np.float32


def _validate_range(value, name):
    if value is None:
        return None
    if len(value) != 2 or not np.all(np.isfinite(value)) or value[0] >= value[1]:
        raise ValueError(f'{name} must contain two finite, strictly increasing values; got {value!r}')
    return float(value[0]), float(value[1])


def _safe_cast(array, dtype):
    dtype = np.dtype(dtype)
    array = np.asarray(array)
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        # Integer casts of NaN and infinity are platform-dependent.
        array = np.nan_to_num(array, nan=0.0, posinf=float(limits.max), neginf=float(limits.min))
        array = np.clip(array, limits.min, limits.max)
    return array.astype(dtype)


def _normalize(array, output_dtype, normalization_range=None, valid_mask=None):
    """Normalize to [0, 1] or the destination integer dtype range."""
    array = np.asarray(array)
    dtype = np.dtype(output_dtype)
    if dtype.kind not in 'fiu':
        raise TypeError(f'Normalization requires a real numeric output dtype, got {dtype}')
    supplied_range = _validate_range(normalization_range, 'normalization_range')
    working = array.astype(np.float64)
    if supplied_range is None:
        selected = working if valid_mask is None else working[valid_mask]
        selected = selected[np.isfinite(selected)]
        if selected.size == 0:
            return np.zeros(array.shape, dtype=np.float64)
        low, high = float(selected.min()), float(selected.max())
        if high == low:
            return np.zeros(array.shape, dtype=np.float64)
    else:
        low, high = supplied_range
    scaled = (np.clip(working, low, high) - low) / (high - low)
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        scaled = scaled * (float(limits.max) - float(limits.min)) + float(limits.min)
    return scaled


def fft_highpass(image: np.ndarray, sigma=(10, 10)) -> np.ndarray:
    """FFT Gaussian high-pass on a 2D periodic image; sigma is (y, x)."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError('fft_highpass expects a 2D image')
    if np.isscalar(sigma):
        sigma = (sigma, sigma)
    if len(sigma) != 2 or not np.all(np.isfinite(sigma)) or np.any(np.asarray(sigma) <= 0):
        raise ValueError('sigma must contain two positive finite values')
    sigma_y, sigma_x = sigma
    ky = np.fft.fftfreq(image.shape[0])[:, None]
    kx = np.fft.fftfreq(image.shape[1])[None, :]
    response = -np.expm1(-2 * np.pi**2 * (sigma_y**2 * ky**2 + sigma_x**2 * kx**2))
    return np.fft.ifft2(np.fft.fft2(image) * response).real


def filter_wrapper(func):
    @wraps(func)
    def wrapper(in_array: np.ndarray, *args, filter_mode='apply', clip=None,
                normalize=False, normalization_range=None, cast_dtype=None,
                keep_zeros=False, zeros_to_mean=False, **kwargs):
        if filter_mode not in ('apply', 'add', 'subtract'):
            raise ValueError(f'Unknown filter_mode: {filter_mode!r}')
        clip = _validate_range(clip, 'clip')
        if normalization_range is not None and not normalize:
            raise ValueError('normalization_range requires normalize=True')
        normalization_range = _validate_range(normalization_range, 'normalization_range')
        original = np.asarray(in_array)
        zero_mask = original == 0 if keep_zeros else None
        if zeros_to_mean:
            nonzero = original != 0
            if np.any(nonzero):
                prepared = np.where(nonzero, original, original[nonzero].mean())
            else:
                prepared = original.copy()
        else:
            prepared = original
        filtered = np.asarray(func(prepared, *args, **kwargs))
        if filtered.shape != original.shape:
            raise ValueError(f'{func.__name__} returned shape {filtered.shape}, expected {original.shape}')
        if filter_mode != 'apply':
            work_dtype = _working_dtype(original, filtered)
            left, right = original.astype(work_dtype), filtered.astype(work_dtype)
            result = left + right if filter_mode == 'add' else left - right
        else:
            result = filtered
        if clip is not None:
            result = np.clip(result, *clip)
        dtype = np.dtype(cast_dtype) if cast_dtype is not None else None
        if normalize:
            target_dtype = dtype if dtype is not None else np.dtype('float32')
            valid_mask = ~zero_mask if zero_mask is not None else None
            result = _normalize(result, target_dtype, normalization_range, valid_mask)
            result = _safe_cast(result, target_dtype)
        elif dtype is not None:
            result = _safe_cast(result, dtype)
        if zero_mask is not None:
            result = np.where(zero_mask, 0, result)
        return result
    return wrapper


class ImageFilter:
    def __init__(self, in_array):
        self._in_array = in_array

    @classmethod
    def validate_pipeline(cls, filters):
        """Validate a JSON-compatible pipeline and return normalized steps.

        Validation is independent of image data. Image-dependent constraints
        (such as input dimensionality) are checked when filters execute.
        """
        if not isinstance(filters, (list, tuple)):
            raise TypeError('Filter pipeline must be a list or tuple of [name, parameters] steps')

        available = cls.get_available_filter_names()
        validated = []
        for index, step in enumerate(filters):
            prefix = f'Filter step {index}'
            if not isinstance(step, (list, tuple)) or len(step) != 2:
                raise ValueError(f'{prefix} must be [filter_name, parameters], got {step!r}')
            name, parameters = step
            if not isinstance(name, str):
                raise TypeError(f'{prefix}: filter name must be a string')
            if name not in available:
                raise ValueError(f'{prefix} ({name!r}): unknown filter; available: {available}')
            if not isinstance(parameters, dict):
                raise TypeError(f'{prefix} ({name!r}): parameters must be a dictionary')
            if not all(isinstance(key, str) for key in parameters):
                raise TypeError(f'{prefix} ({name!r}): parameter names must be strings')

            method = getattr(cls, name)
            # @wraps preserves the underlying filter signature. The wrapper
            # accepts a separate set of common processing parameters.
            filter_signature = inspect.signature(method)
            wrapper_signature = inspect.signature(method, follow_wrapped=False)
            wrapper_keys = set(wrapper_signature.parameters) - {'in_array', 'args', 'kwargs'}
            filter_keys = set(filter_signature.parameters) - {'in_array'}
            unknown = set(parameters) - wrapper_keys - filter_keys
            if unknown:
                raise ValueError(f'{prefix} ({name!r}): unknown parameter(s) {sorted(unknown)}; '
                                 f'allowed: {sorted(wrapper_keys | filter_keys)}')
            try:
                filter_signature.bind(np.empty((0,)), **{
                    key: value for key, value in parameters.items() if key in filter_keys
                })
            except TypeError as exc:
                raise ValueError(f'{prefix} ({name!r}): {exc}') from exc

            mode = parameters.get('filter_mode', 'apply')
            if mode not in ('apply', 'add', 'subtract'):
                raise ValueError(f'{prefix} ({name!r}): invalid filter_mode {mode!r}')
            if parameters.get('normalization_range') is not None and not parameters.get('normalize', False):
                raise ValueError(f'{prefix} ({name!r}): normalization_range requires normalize=True')
            for key in ('clip', 'normalization_range'):
                if parameters.get(key) is not None:
                    try:
                        _validate_range(parameters[key], key)
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f'{prefix} ({name!r}): {exc}') from exc
            if parameters.get('cast_dtype') is not None:
                try:
                    np.dtype(parameters['cast_dtype'])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'{prefix} ({name!r}): invalid cast_dtype: {exc}') from exc
            validated.append((name, parameters.copy()))
        return validated

    def get_filtered(self, filters, in_array=None, return_intermediates=False, verbose=False):
        steps = self.validate_pipeline(filters)
        result = np.asarray(self._in_array).copy() if in_array is None else np.asarray(in_array)
        intermediates = []
        for index, (name, kwargs) in enumerate(steps):
            if verbose:
                print(f'Applying filter {name} with kwargs {kwargs} ...')
            try:
                result = getattr(self, name)(result, **kwargs)
            except (ValueError, TypeError) as exc:
                raise type(exc)(f'Filter step {index} ({name!r}): {exc}') from exc
            if return_intermediates:
                intermediates.append(result.copy())
        return (result, intermediates) if return_intermediates else result

    def get_filtered_stack(self, filters, n_workers=1):
        steps = self.validate_pipeline(filters)
        if not (2 < self._in_array.ndim <= 4):
            raise RuntimeError('Stack processing requires 3D or 4D input')
        if n_workers == 1:
            return np.asarray([self.get_filtered(steps, in_array=sl) for sl in self._in_array])
        with Pool(processes=n_workers) as pool:
            return np.asarray(pool.starmap(_filter_slice, [(sl, steps) for sl in self._in_array]))

    @classmethod
    def get_available_filter_names(cls):
        return [name for name, member in inspect.getmembers(cls, inspect.isfunction)
                if not name.startswith('_') and not name.startswith('get_')]

    @staticmethod
    @filter_wrapper
    def gaussian(in_array, sigma=1.0):
        from vigra.filters import gaussianSmoothing
        return gaussianSmoothing(in_array.astype(_working_dtype(in_array)), sigma=sigma)

    @staticmethod
    @filter_wrapper
    def gaussian_gradient_magnitude(in_array, sigma=1.0):
        from vigra.filters import gaussianGradientMagnitude
        return gaussianGradientMagnitude(in_array.astype(_working_dtype(in_array)), sigma=sigma)

    @staticmethod
    @filter_wrapper
    def median(in_array, radius=None, footprint=None, elliptical_footprint=False,
               on_binned=1, sample_footprint=None, random_seed=None):
        from scipy.ndimage import median_filter, zoom
        if (radius is None) == (footprint is None):
            raise ValueError('Supply exactly one of radius and footprint')
        if not isinstance(on_binned, (int, np.integer)) or on_binned < 1:
            raise ValueError('on_binned must be a positive integer')
        if in_array.ndim != 2:
            raise ValueError('median currently expects a 2D image')
        original_shape = in_array.shape
        if on_binned != 1:
            from squirrel.library.scaling import average_bin_image
            in_array = average_bin_image(in_array, on_binned)
        if radius is not None:
            if np.isscalar(radius):
                radius = (radius, radius)
            if len(radius) != 2 or any(not isinstance(r, (int, np.integer)) or r < 0 for r in radius):
                raise ValueError('radius must be nonnegative integer or pair of nonnegative integers')
            ry, rx = radius
            if elliptical_footprint:
                y, x = np.ogrid[-ry:ry + 1, -rx:rx + 1]
                footprint = ((y / max(ry, 1))**2 + (x / max(rx, 1))**2 <= 1)
            elif sample_footprint is not None:
                footprint = np.ones((2 * ry + 1, 2 * rx + 1), dtype=bool)
            else:
                filtered = median_filter(in_array, size=(2 * ry + 1, 2 * rx + 1))
                footprint = None
        if footprint is not None:
            footprint = np.asarray(footprint, dtype=bool)
            if footprint.ndim != 2 or not np.any(footprint):
                raise ValueError('footprint must be a nonempty 2D boolean mask')
            if sample_footprint is not None:
                if not isinstance(sample_footprint, (int, np.integer)) or sample_footprint < 1:
                    raise ValueError('sample_footprint must be a positive integer')
                coords = np.argwhere(footprint)
                if sample_footprint < len(coords):
                    rng = np.random.default_rng(random_seed)
                    chosen = coords[rng.choice(len(coords), size=sample_footprint, replace=False)]
                    footprint = np.zeros_like(footprint)
                    footprint[tuple(chosen.T)] = True
            from skimage.filters import median as sk_median
            filtered = sk_median(in_array, footprint=footprint)
        if on_binned != 1:
            # Explicit target shape avoids unfilled zero borders for nondivisible dimensions.
            factors = [target / source for target, source in zip(original_shape, filtered.shape)]
            filtered = zoom(filtered, factors, order=3, mode='nearest', grid_mode=True)
            filtered = filtered[tuple(slice(0, n) for n in original_shape)]
        return filtered

    @staticmethod
    @filter_wrapper
    def clahe(in_array, clip_limit=3.0, tile_grid_size=(127, 127), invert_output=False,
              auto_mask=False, background_to_mean=False, tile_grid_in_pixels=False):
        from squirrel.library.normalization import clahe_on_image
        return clahe_on_image(in_array, clip_limit=clip_limit, tile_grid_size=tile_grid_size,
                              invert_output=invert_output, auto_mask=auto_mask,
                              background_to_mean=background_to_mean, tile_grid_in_pixels=tile_grid_in_pixels)

    @staticmethod
    @filter_wrapper
    def vsnr(in_array, filters=None, is_gpu=True, maxit=20, algo='auto'):
        from squirrel.library.vsnr import vsnr_on_image
        return vsnr_on_image(in_array, filters=filters, is_gpu=is_gpu, maxit=maxit, algo=algo)

    @staticmethod
    @filter_wrapper
    def fft_highpass(in_array, sigma=(10, 10)):
        return fft_highpass(in_array, sigma)

    @staticmethod
    @filter_wrapper
    def invert(in_array, invert_range=None):
        if invert_range is not None:
            low, high = _validate_range(invert_range, 'invert_range')
            return (low + high) - in_array.astype(_working_dtype(in_array))
        if np.issubdtype(in_array.dtype, np.unsignedinteger):
            return np.iinfo(in_array.dtype).max - in_array
        raise ValueError('invert_range is required for signed integer or floating-point images')

    @staticmethod
    @filter_wrapper
    def same(in_array):
        return in_array


def _filter_slice(image, filters):
    return ImageFilter(image).get_filtered(filters)
