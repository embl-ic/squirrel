"""Image filtering and JSON-compatible filter pipelines."""
import inspect
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
from multiprocessing import Pool, get_context

import numpy as np

from squirrel.library.normalization import normalize_image, _validate_pair
from squirrel.library.dtype import working_dtype, safe_cast, rescale_integer_dtype


def _validate_range(value, name):
    if value is None:
        return None
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError(f'{name} must be a sequence of two real numbers, got {value!r}')
    try:
        values = np.asarray(value)
        if values.ndim != 1 or values.size != 2:
            raise ValueError(f'{name} must contain exactly two values, got {value!r}')
        if values.dtype.kind not in 'iuf':
            raise TypeError(f'{name} must contain real numbers, got {value!r}')
        low, high = float(values[0]), float(values[1])
    except (OverflowError, TypeError, ValueError) as exc:
        if isinstance(exc, (TypeError, ValueError)) and str(exc).startswith(name):
            raise
        raise TypeError(f'{name} must contain real numbers, got {value!r}') from exc
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError(f'{name} must contain two finite, strictly increasing values; got {value!r}')
    return low, high


def _validate_boolean_options(parameters, prefix=''):
    """Validate common boolean options without coercing truthy values."""
    for name in ('normalize', 'keep_zeros', 'ignore_zeros', 'zeros_to_mean'):
        if name in parameters and not isinstance(parameters[name], (bool, np.bool_)):
            raise TypeError(f'{prefix}{name} must be a boolean, got {parameters[name]!r}')



def _validate_processing_options(parameters):
    """Validate common image-independent options for direct and pipeline calls.

    Return a copy with validated ranges, leaving caller parameters unchanged.
    """
    options = parameters.copy()
    _validate_boolean_options(options)

    mode = options.get('filter_mode', 'apply')
    if not isinstance(mode, str):
        raise TypeError(f'filter_mode must be a string, got {mode!r}')
    if mode not in ('apply', 'add', 'subtract'):
        raise ValueError(f'invalid filter_mode {mode!r}')

    normalize = options.get('normalize', False)
    if options.get('normalization_range') is not None and not normalize:
        raise ValueError('normalization_range requires normalize=True')
    for key in ('clip', 'normalization_range'):
        if options.get(key) is not None:
            options[key] = _validate_range(options[key], key)

    method = options.get('normalization_method', 'minmax')
    if method not in ('minmax', 'quantile'):
        raise ValueError(f'Unknown normalization_method: {method!r}')
    quantiles = options.get('quantiles', (0.01, 0.99))
    anchors = options.get('anchors', (0.0, 1.0))
    _validate_pair(quantiles, 'quantiles', unit=True)
    _validate_pair(anchors, 'anchors', unit=True)
    if not normalize and (method != 'minmax' or tuple(quantiles) != (0.01, 0.99) or
                          tuple(anchors) != (0.0, 1.0)):
        raise ValueError('normalization options require normalize=True')

    rescale_dtype = options.get('rescale_dtype')
    if rescale_dtype is not None:
        if normalize:
            raise ValueError('rescale_dtype cannot be used with normalize=True; '
                             'remove normalize=True or rescale_dtype.')
        if options.get('cast_dtype') is not None:
            raise ValueError('rescale_dtype cannot be used with cast_dtype; remove one of these options.')
        try:
            target_dtype = np.dtype(rescale_dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'invalid rescale_dtype {rescale_dtype!r}; '
                             'choose "uint8" or "uint16".') from exc
        if target_dtype not in (np.dtype('uint8'), np.dtype('uint16')):
            raise TypeError(f'rescale_dtype must be uint8 or uint16, got {rescale_dtype!r}; '
                            'choose "uint8" or "uint16".')

    if options.get('cast_dtype') is not None:
        try:
            np.dtype(options['cast_dtype'])
        except (TypeError, ValueError) as exc:
            raise ValueError(f'invalid cast_dtype: {exc}') from exc
    return options


def filter_wrapper(func):
    """Decorate an image filter with shared processing and validation options.

    The wrapped callable accepts ``filter_mode`` (apply/add/subtract),
    ``clip``, ``normalize``, ``normalization_range``, ``cast_dtype``,
    ``rescale_dtype``,
    ``normalization_method``, ``quantiles``, ``anchors``,
    ``keep_zeros``, ``ignore_zeros``, and ``zeros_to_mean`` in addition to filter-specific
    arguments. The result retains the input shape; invalid options raise
    ``TypeError`` or ``ValueError``.
    """
    @wraps(func)
    def wrapper(in_array: np.ndarray, *args, filter_mode='apply', clip=None,
                normalize=False, normalization_range=None, cast_dtype=None,
                 rescale_dtype=None,
                normalization_method='minmax', quantiles=(0.01, 0.99),
                anchors=(0.0, 1.0),
                keep_zeros=False, ignore_zeros=False, zeros_to_mean=False, **kwargs):
        options = dict(filter_mode=filter_mode, clip=clip, normalize=normalize,
                       normalization_range=normalization_range, cast_dtype=cast_dtype,
                       rescale_dtype=rescale_dtype, normalization_method=normalization_method,
                       quantiles=quantiles, anchors=anchors, keep_zeros=keep_zeros,
                       ignore_zeros=ignore_zeros, zeros_to_mean=zeros_to_mean)
        try:
            validated_options = _validate_processing_options(options)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"Filter {func.__name__!r}: {exc}") from exc
        clip = validated_options['clip']
        normalization_range = validated_options['normalization_range']
        # Apply the same image-independent, filter-specific checks to direct
        # calls as to JSON pipelines (including positional filter arguments).
        bound = inspect.signature(func).bind(in_array, *args, **kwargs)
        filter_parameters = {key: value for key, value in bound.arguments.items()
                             if key != 'in_array'}
        try:
            ImageFilter.validate_pipeline([[func.__name__, filter_parameters]])
        except (TypeError, ValueError) as exc:
            # Direct calls have no pipeline index: identify the filter instead.
            prefix = f"Filter step 0 ({func.__name__!r}): "
            detail = str(exc).removeprefix(prefix)
            raise type(exc)(f"Filter {func.__name__!r}: {detail}") from exc
        original = np.asarray(in_array)
        if original.size == 0:
            raise ValueError(f"Filter {func.__name__!r}: input image must not be empty")
        if original.dtype.kind not in 'iuf':
            raise TypeError(f"Filter {func.__name__!r}: input image must have a real numeric dtype, "
                            f"got {original.dtype}")
        zero_mask = original == 0 if keep_zeros else None
        if zeros_to_mean:
            nonzero = original != 0
            if np.any(nonzero):
                mean_value = original[nonzero].mean()

                if np.issubdtype(original.dtype, np.integer):
                    mean_value = np.asarray(mean_value).astype(original.dtype)[()]

                prepared = np.where(nonzero, original, mean_value)
            else:
                prepared = original.copy()
        else:
            prepared = original
        try:
            filtered = np.asarray(func(prepared, *args, **kwargs))
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"Filter {func.__name__!r}: {exc}") from exc
        if filtered.shape != original.shape:
            raise ValueError(f'{func.__name__} returned shape {filtered.shape}, expected {original.shape}')
        if filter_mode != 'apply':
            work_dtype = working_dtype(original, filtered)
            left, right = original.astype(work_dtype), filtered.astype(work_dtype)
            result = left + right if filter_mode == 'add' else left - right
        else:
            result = filtered
        if clip is not None:
            result = np.clip(result, *clip)
        dtype = np.dtype(cast_dtype) if cast_dtype is not None else None
        if rescale_dtype is not None:
            result = rescale_integer_dtype(result, rescale_dtype)
        elif normalize:
            target_dtype = dtype if dtype is not None else np.dtype('float32')
            result = normalize_image(
                result, method=normalization_method, quantiles=quantiles,
                normalization_range=normalization_range, anchors=anchors,
                output_dtype=target_dtype, ignore_zeros=ignore_zeros)
        elif dtype is not None:
            result = safe_cast(result, dtype)
        if zero_mask is not None:
            result = np.where(zero_mask, 0, result)
        return result
    return wrapper


class ImageFilter:
    def __init__(self, in_array: np.ndarray):
        """Create a filter executor for an image or stack.

        Parameters
        ----------
        in_array : numpy.ndarray
            Source image for ``get_filtered`` or stack for
            ``get_filtered_stack``. Input validation occurs at execution.
        """
        self._in_array = in_array

    @classmethod
    def validate_pipeline(cls, filters: list | tuple) -> list[tuple[str, dict]]:
        """Validate a JSON-compatible sequence of filter steps.

        Parameters
        ----------
        filters : list or tuple
            Sequence of ``[filter_name, parameter_dict]`` pairs. Filter
            names must be available on :class:`ImageFilter`.

        Returns
        -------
        list of tuple[str, dict]
            Validated ``(name, copied_parameters)`` pairs.

        Raises
        ------
        TypeError
            A pipeline, name, or parameter has an invalid type.
        ValueError
            A step is malformed or a parameter value is invalid.

        Notes
        -----
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

            try:
                _validate_processing_options(parameters)
            except (TypeError, ValueError) as exc:
                raise type(exc)(f'{prefix} ({name!r}): {exc}') from exc
            # Validate image-independent filter parameters before processing any data.
            try:
                if name in ('gaussian', 'gaussian_gradient_magnitude', 'fft_highpass'):
                    sigma = np.asarray(parameters.get('sigma', 1.0 if name != 'fft_highpass' else (10, 10)))
                    if sigma.ndim > 1 or sigma.size not in (1, 2):
                        raise ValueError('sigma must be a scalar or a pair of positive finite numbers')
                    if not np.issubdtype(sigma.dtype, np.number) or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
                        raise ValueError('sigma must contain positive finite numbers')
                elif name == 'median':
                    radius = parameters.get('radius')
                    footprint = parameters.get('footprint')
                    if (radius is None) == (footprint is None):
                        raise ValueError('supply exactly one of radius and footprint')
                    if radius is not None:
                        radii = (radius,) if np.isscalar(radius) else radius
                        if not isinstance(radii, (tuple, list, np.ndarray)) or len(radii) not in (1, 2) or any(
                            not isinstance(r, (int, np.integer)) or isinstance(r, (bool, np.bool_)) or r < 0
                            for r in radii
                        ):
                            raise ValueError('radius must be a nonnegative integer or pair of nonnegative integers')
                    else:
                        mask = np.asarray(footprint)
                        if mask.ndim != 2 or not np.any(mask):
                            raise ValueError('footprint must be a nonempty 2D mask')
                    for key in ('on_binned', 'sample_footprint'):
                        value = parameters.get(key, 1 if key == 'on_binned' else None)
                        if value is not None and (not isinstance(value, (int, np.integer)) or
                                                  isinstance(value, (bool, np.bool_)) or value < 1):
                            raise ValueError(f'{key} must be a positive integer')
                    seed = parameters.get('random_seed')
                    if seed is not None and (not isinstance(seed, (int, np.integer)) or
                                             isinstance(seed, (bool, np.bool_)) or seed < 0):
                        raise ValueError('random_seed must be a nonnegative integer or None')
                elif name == 'invert' and parameters.get('invert_range') is not None:
                    _validate_range(parameters['invert_range'], 'invert_range')
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{prefix} ({name!r}): {exc}') from exc
            validated.append((name, parameters.copy()))
        return validated

    def get_filtered(self, filters: list | tuple, in_array: np.ndarray | None = None,
                     return_intermediates: bool = False, verbose: bool = False
                     ) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
        """Apply a validated pipeline to one image.

        Parameters
        ----------
        filters : list or tuple
            Pipeline of ``[filter_name, parameter_dict]`` steps.
        in_array : numpy.ndarray or None, optional
            Override the image supplied at initialization.
        return_intermediates : bool, optional
            Return a copy of the result after each pipeline step.
        verbose : bool, optional
            Print each step before execution.

        Returns
        -------
        numpy.ndarray or tuple[numpy.ndarray, list[numpy.ndarray]]
            Filtered image, optionally accompanied by intermediate images.

        Raises
        ------
        TypeError, ValueError
            If validation or a filter execution fails. Execution errors
            include the failing pipeline step index and filter name.
        """
        # These options control the return contract and logging; do not
        # silently interpret integers or strings as booleans.
        for name, value in (('return_intermediates', return_intermediates), ('verbose', verbose)):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f'{name} must be a boolean, got {value!r}')
        steps = self.validate_pipeline(filters)
        result = np.asarray(self._in_array).copy() if in_array is None else np.asarray(in_array)
        intermediates = []
        for index, (name, kwargs) in enumerate(steps):
            if verbose:
                print(f'Applying filter {name} with kwargs {kwargs} ...')
            try:
                result = getattr(self, name)(result, **kwargs)
            except (ValueError, TypeError) as exc:
                # A direct filter call already identifies the filter; avoid
                # repeating it when adding the pipeline step context.
                detail = str(exc).removeprefix(f"Filter {name!r}: ")
                raise type(exc)(f'Filter step {index} ({name!r}): {detail}') from exc
            if return_intermediates:
                intermediates.append(result.copy())
        return (result, intermediates) if return_intermediates else result

    def get_filtered_stack(self, filters: list | tuple, n_workers: int = 1,
                           backend: str = 'threading', progress: bool = True,
                           progress_callback=None, return_intermediates: bool = False) -> np.ndarray:
        """Apply the same pipeline independently to each stack slice.

        Parameters
        ----------
        filters : list or tuple
            Pipeline of ``[filter_name, parameter_dict]`` steps.
        n_workers : int, optional
            Positive number of workers. One runs serially.
        backend : str, optional
            ``"threading"`` (default), ``"multiprocessing"``, or ``"serial"``.
            The serial backend requires ``n_workers=1``.
        progress : bool, optional
            Show a slice-level progress bar when no callback is supplied.
        return_intermediates : bool, optional
            Return filtered stack and per-step intermediate stacks.
        progress_callback : callable or None, optional
            Called with the number of newly completed slices. Allows workflows
            to maintain a single progress bar across batches.

        Returns
        -------
        numpy.ndarray
            Stack of filtered slices in their original order.

        Raises
        ------
        TypeError, ValueError
            If the pipeline is invalid, the input is not a nonempty 3D or
            4D stack, ``n_workers`` is not a positive integer, or the
            execution backend is invalid.
        """
        steps = self.validate_pipeline(filters)
        if isinstance(n_workers, (bool, np.bool_)) or not isinstance(n_workers, (int, np.integer)) or n_workers < 1:
            raise ValueError(f'n_workers must be a positive integer, got {n_workers!r}')
        if not isinstance(backend, str):
            raise TypeError(f'backend must be a string, got {backend!r}')
        if backend not in ('serial', 'multiprocessing', 'threading'):
            raise ValueError(f'Unknown filtering backend {backend!r}; expected '
                             "'serial', 'multiprocessing', or 'threading'")
        if backend == 'serial' and n_workers != 1:
            raise ValueError('backend="serial" requires n_workers=1')
        stack = np.asarray(self._in_array)
        if stack.ndim not in (3, 4):
            raise ValueError(f'Stack processing requires 3D or 4D input, got {stack.ndim}D')
        if stack.size == 0 or stack.shape[0] == 0:
            raise ValueError('Stack processing requires a nonempty input stack')
        if not isinstance(progress, (bool, np.bool_)):
            raise TypeError(f'progress must be a boolean, got {progress!r}')
        if not isinstance(return_intermediates, (bool, np.bool_)):
            raise TypeError(f'return_intermediates must be a boolean, got {return_intermediates!r}')
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError('progress_callback must be callable or None')

        from tqdm.auto import tqdm

        with (tqdm(total=stack.shape[0], desc='Filtering', unit='slice')
              if progress and progress_callback is None else nullcontext()) as bar:
            def completed_slices(results):
                for result in results:
                    if progress_callback is not None:
                        progress_callback(1)
                    elif bar is not None:
                        bar.update(1)
                    yield result

            def assemble(results):
                results = list(completed_slices(results))
                if not return_intermediates:
                    return np.asarray(results)
                filtered, intermediates = zip(*results)
                return np.asarray(filtered), [
                    np.asarray(step) for step in zip(*intermediates)
                ] if steps else []

            if n_workers == 1:
                results = (self.get_filtered(steps, in_array=sl,
                            return_intermediates=return_intermediates) for sl in stack)
                return assemble(results)

            worker = (partial(_filter_slice, filters=steps, return_intermediates=True)
                      if return_intermediates else partial(_filter_slice, filters=steps))
            if backend == 'threading':
                with ThreadPoolExecutor(max_workers=int(n_workers)) as executor:
                    return assemble(executor.map(worker, stack))

            # CLAHE uses OpenCV, whose native thread state can be unsafe to
            # inherit with fork after CLAHE has run in the parent process.
            if any(name == 'clahe' for name, _ in steps):
                with get_context('spawn').Pool(
                        processes=int(n_workers), initializer=_init_opencv_worker) as pool:
                    return assemble(pool.imap(worker, stack, chunksize=1))

            with Pool(processes=int(n_workers)) as pool:
                return assemble(pool.imap(worker, stack, chunksize=1))

    @classmethod
    def get_available_filter_names(cls) -> list[str]:
        """Return names of publicly available image filter methods."""
        return [name for name, member in inspect.getmembers(cls, inspect.isfunction)
                if not name.startswith('_') and not name.startswith('get_')]

    @staticmethod
    @filter_wrapper
    def gaussian(in_array, sigma=1.0):
        from vigra.filters import gaussianSmoothing
        return gaussianSmoothing(in_array.astype(working_dtype(in_array)), sigma=sigma)

    @staticmethod
    @filter_wrapper
    def gaussian_gradient_magnitude(in_array, sigma=1.0):
        from vigra.filters import gaussianGradientMagnitude
        return gaussianGradientMagnitude(in_array.astype(working_dtype(in_array)), sigma=sigma)

    @staticmethod
    @filter_wrapper
    def median(in_array, radius=None, footprint=None, elliptical_footprint=False,
               on_binned=1, sample_footprint=None, random_seed=None):
        from squirrel.library.image import median_filter_2d
        return median_filter_2d(in_array, radius=radius, footprint=footprint,
                                elliptical_footprint=elliptical_footprint,
                                on_binned=on_binned, sample_footprint=sample_footprint,
                                random_seed=random_seed)

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
        from squirrel.library.image import fft_highpass
        return fft_highpass(in_array, sigma)

    @staticmethod
    @filter_wrapper
    def invert(in_array, invert_range=None):
        if invert_range is not None:
            low, high = _validate_range(invert_range, 'invert_range')
            return (low + high) - in_array.astype(working_dtype(in_array))
        if np.issubdtype(in_array.dtype, np.unsignedinteger):
            return np.iinfo(in_array.dtype).max - in_array
        raise ValueError('invert_range is required for signed integer or floating-point images')

    @staticmethod
    @filter_wrapper
    def same(in_array):
        return in_array


def _filter_slice(image, filters, return_intermediates=False):
    if return_intermediates:
        return ImageFilter(image).get_filtered(filters, return_intermediates=True)
    return ImageFilter(image).get_filtered(filters)


def _init_opencv_worker():
    """Limit OpenCV native threads only within a multiprocessing worker."""
    import cv2
    cv2.setNumThreads(1)
