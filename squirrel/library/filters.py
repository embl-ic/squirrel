
import numpy as np
import inspect
from functools import wraps


def fft_highpass(
        image: np.ndarray,
        sigma: tuple[float, float] = (10, 10)
) -> np.ndarray:
    """
    Apply an anisotropic high-pass filter to a 2D image using FFT,
    matching a Gaussian high-pass filter with given spatial sigmas.

    Parameters
    ----------
    image : 2D np.ndarray
        Input image
    sigma : float
        Gaussian sigmas along x- and y-axis (pixels)

    Returns
    -------
    result : 2D np.ndarray
        High-pass filtered image
    """

    sigma_y, sigma_x = sigma
    ny, nx = image.shape

    # FFT of the image
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)  # center zero frequency

    # Frequency grids (cycles per pixel, shifted to [-0.5, 0.5))
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    KX, KY = np.meshgrid(kx, ky)

    # Convert spatial sigma to frequency domain parameter
    sx_freq = nx / (2 * np.pi * sigma_x)
    sy_freq = ny / (2 * np.pi * sigma_y)

    # Anisotropic high-pass filter
    H = 1 - np.exp(-((KX * nx)**2 / sx_freq**2 + (KY * ny)**2 / sy_freq**2))

    # Apply filter
    F_filtered = F_shifted * H

    # Transform back to spatial domain
    result = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))

    return result


def filter_wrapper(func):
    @wraps(func)
    def wrapper(in_array: np.ndarray, *args,
                filter_mode: str = 'apply',
                clip: tuple[float, float] | None = None,
                cast_dtype: np.dtype | None = None,
                keep_zeros: bool = False,
                **kwargs) -> np.ndarray:

        if filter_mode not in ['apply', 'add', 'subtract']:
            raise ValueError(f'Invalid value for filter_mode: {filter_mode}; Valid values: ["apply", "add", "subtract"]')
        if clip is not None and len(clip) != 2:
            raise ValueError(f"clip must have exactly two values (low, high); instead got: {clip}")
        
        cast_dtype = np.dtype(cast_dtype) if cast_dtype is not None else None

        original = in_array
        result = func(in_array, *args, **kwargs)

        if filter_mode == 'add':
            result = original.astype('float32') + result.astype('float32')
        if filter_mode == 'subtract':
            result = original.astype('float32') - result.astype('float32')

        if clip is not None:
            result = np.clip(result, *clip)

        if np.issubdtype(cast_dtype, np.unsignedinteger):
            result -= result.min()
            result = result / result.max() * np.iinfo(cast_dtype).max
            result = result.astype(cast_dtype)
        elif cast_dtype is not None:
            result = result.astype(cast_dtype)

        if keep_zeros:
            result = np.where(original == 0, 0, result)

        return result

    return wrapper


class ImageFilter:

    def __init__(self, in_array):
        self._in_array = in_array

    def _check_filter_names(self, filter_names):
        available_filters = self.get_available_filter_names()
        for filter_name in filter_names:
            if filter_name not in available_filters:
                raise ValueError(f'{filter_name} not in available filters: {available_filters}')

    def get_filtered(
            self,
            filters: list[list[object]],
            in_array: np.ndarray = None,
            return_intermediates: bool = False
    ) -> np.ndarray:

        filter_names = [filter[0] for filter in filters]
        self._check_filter_names(filter_names)

        intermediates = []

        result_array = self._in_array.copy() if in_array is None else in_array
        for filter_name, filter_kwargs in filters:
            # print(f'Applying filter {filter_name} with kwargs {filter_kwargs} ...')
            result_array = getattr(self, filter_name)(result_array, **filter_kwargs)
            if return_intermediates:
                intermediates.append(result_array)

        if return_intermediates:
            return result_array, intermediates
        return result_array

    def get_filtered_stack(
            self, filters: list[list[object]], n_workers: int = 1
    ) -> np.ndarray:
        if not (2 < self._in_array.ndim <= 4):
            raise RuntimeError(f'in_array.ndim = {self._in_array.ndim}. \n'
                               'in_array needs to have ndim=3 (stack of 2D images) or ndim=4 (stack of 3D images) '
                               'for stack processing.')

        if n_workers == 1:
            result_array = np.array([
                self.get_filtered(filters, in_array=im_slice)
                for im_slice in self._in_array
            ])

        else:
            print(f'Running ImageFilter stack with {n_workers} CPUs')
            from multiprocessing import Pool
            with Pool(processes=n_workers) as p:
                tasks = [
                    p.apply_async(
                        self.get_filtered, (filters,), dict(in_array=im_slice)
                    )
                    for im_slice in self._in_array
                ]
                result_array = np.array([task.get() for task in tasks])

        return result_array

    def get_available_filter_names(self):
        exclude = []
        return [
            name for name, member in inspect.getmembers(self.__class__, predicate=inspect.isfunction)
            if not name.startswith('_') and not name.startswith('get_') and name not in exclude
        ]

    @staticmethod
    @filter_wrapper
    def gaussian(in_array: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        from vigra.filters import gaussianSmoothing
        if in_array.dtype == 'uint16':
            in_array = in_array.astype('float32')
            return gaussianSmoothing(in_array, sigma=sigma).astype('uint16')

        return gaussianSmoothing(in_array, sigma=sigma)

    @staticmethod
    @filter_wrapper
    def gaussian_gradient_magnitude(in_array: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        from vigra.filters import gaussianGradientMagnitude
        return gaussianGradientMagnitude(in_array.astype('float32'), sigma=sigma)

    @staticmethod
    @filter_wrapper
    def median(
            in_array: np.ndarray,
            radius: int | tuple[int, int] | None = None,
            footprint: np.ndarray | None = None,
            elliptical_footprint: bool = False,
            on_binned: int = 1,
            sample_footprint: int = None
    ):
        def _sample_footprint(footprint, n_samples=100):
            print(f'sampling footprint with {n_samples} samples ...')

            # Get coordinates of True pixels
            true_coords = np.argwhere(footprint)

            # If requested samples > available pixels, return original
            if n_samples >= len(true_coords):
                return footprint

            # Randomly choose indices
            chosen_idx = np.random.choice(len(true_coords), size=n_samples, replace=False)
            selected_coords = true_coords[chosen_idx]

            # Build new footprint
            subsampled = np.zeros_like(footprint, dtype=bool)
            subsampled[tuple(selected_coords.T)] = True

            return subsampled

        if not (bool(radius is None) ^ bool(footprint is None)):
            raise ValueError('Supply either radius or footprint, and not both!')

        shape = in_array.shape
        if on_binned != 1:
            from squirrel.library.scaling import average_bin_image
            in_array = average_bin_image(img, on_binned)

        if radius is not None:
            # Normalize radius to a tuple
            if isinstance(radius, int):
                radius = (radius, radius)

            # Create an elliptical or rectangular footprint
            if elliptical_footprint:
                print(f'computing median with skimage and elliptical footprint ...')
                y, x = np.ogrid[-radius[0]:radius[0] + 1, -radius[1]:radius[1] + 1]
                mask = (x ** 2 / radius[1] ** 2 + y ** 2 / radius[0] ** 2) <= 1
                footprint = mask.astype(bool)
                if sample_footprint is not None:
                    footprint = _sample_footprint(footprint, sample_footprint)
                # Only possible with skimage
                from skimage.filters import median
                in_array = median(in_array, footprint=footprint)
            elif sample_footprint is None:
                print(f'computing median with scipy ...')
                # Supposedly faster with scipy
                from scipy.ndimage import median_filter
                in_array = median_filter(in_array, size=(2 * radius[0] + 1, 2 * radius[1] + 1))
            else:
                print(f'computing median with skimage and rectangular footprint ...')
                footprint_shape = (2 * radius[0] + 1, 2 * radius[1] + 1)
                footprint = np.ones(footprint_shape, dtype=bool)
                footprint = _sample_footprint(footprint, sample_footprint)
                from skimage.filters import median
                in_array = median(in_array, footprint=footprint)

        elif footprint is not None:
            print(f'computing median with skimage and custom footprint ...')
            if sample_footprint is not None:
                footprint = _sample_footprint(footprint, sample_footprint)
            from skimage.filters import median
            in_array = median(in_array, footprint=footprint)

        if on_binned != 1:
            from squirrel.library.scaling import scale_image
            return_img = np.zeros(shape, dtype=in_array.dtype)
            return_img[:in_array.shape[0] * on_binned, :in_array.shape[1] * on_binned] = scale_image(in_array, on_binned, order=3)
        else:
            return_img = in_array

        return return_img

    @staticmethod
    @filter_wrapper
    def clahe(
            in_array: np.ndarray,
            clip_limit: float = 3.0,
            tile_grid_size: tuple[int, int] = (127, 127),
            # cast_dtype: str = None,
            invert_output: bool = False,
            # gaussian_sigma: float = 0.0,
            auto_mask: bool = False,
            background_to_mean: bool = False,
            tile_grid_in_pixels: bool = False
    ) -> np.ndarray:
        from squirrel.library.normalization import clahe_on_image
        return clahe_on_image(
            in_array,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            # cast_dtype=cast_dtype,
            invert_output=invert_output,
            # gaussian_sigma=gaussian_sigma,
            auto_mask=auto_mask,
            background_to_mean=background_to_mean,
            tile_grid_in_pixels=tile_grid_in_pixels
        )

    @staticmethod
    @filter_wrapper
    def vsnr(
            in_array: np.ndarray,
            filters: list[dict] = None,
            is_gpu: bool = True,
            maxit: int = 20,
            algo: str = 'auto'
    ) -> np.ndarray:
        from squirrel.library.vsnr import vsnr_on_image
        return vsnr_on_image(
            in_array,
            filters=filters,
            is_gpu=is_gpu,
            maxit=maxit,
            algo=algo
        )

    @staticmethod
    @filter_wrapper
    def fft_highpass(
            in_array: np.ndarray,
            sigma: tuple[float, float] = (10, 10)
    ) -> np.ndarray:
        return fft_highpass(in_array, sigma)
    
    @staticmethod
    @filter_wrapper
    def invert(in_array: np.ndarray) -> np.ndarray:
        return np.iinfo(in_array.dtype).max - in_array
    
    @staticmethod
    @filter_wrapper
    def same(in_array: np.ndarray) -> np.ndarray:
        return in_array
    

if __name__ == '__main__':

    # img = np.random.randint(0, 128, (200, 200), dtype=np.uint8)
    # imf = ImageFilter(img)
    # # imf = ImageFilter(np.zeros([10, 100, 100]))
    # print(imf.get_available_filter_names())
    # # imf.get_filtered(['gaussian', 'gaussian_gradient_magnitude'], [dict(sigma=1.0), dict(sigma=1.0)])
    # # imf.get_filtered_batch(['gaussian'], [dict(sigma=1.0)], n_workers=10)
    # result = imf.get_filtered(['clahe'], [dict(tile_grid_in_pixels=True, tile_grid_size=(31, 31))])
    #
    # from matplotlib import pyplot as plt
    # plt.imshow(np.concatenate([img, result], axis=1), cmap='gray')
    # plt.show()

    from squirrel.library.io import read_tif_slice
    img = read_tif_slice(filepath='/media/julian/Data/tmp/clahe_test/image.tif', return_filepath=False)

    imf = ImageFilter(img)

    # result = imf.get_filtered(
    #     [
    #         ['vsnr', dict(is_gpu=True, maxit=100, filters=[
    #             dict(name='Gabor', sigma=[2, 35], theta=0, noise_level=0.5),  #, frequency=0.3),
    #         ])],
    #         ['vsnr', dict(is_gpu=True, maxit=100, filters=[
    #             dict(name='Gabor', sigma=[2, 35], theta=90, noise_level=0.5),
    #         ])],
    #         ['vsnr', dict(is_gpu=True, maxit=100, filters=[
    #             dict(name='Gabor', sigma=[5, 35], theta=90, noise_level=0.5),
    #         ])],
    #         ['clahe', dict(tile_grid_in_pixels=True, tile_grid_size=(63, 63))],
    #         ['gaussian', dict(sigma=1.5)]
    #     ]
    # )

    result, intermediates = imf.get_filtered(
        [
            ['vsnr', dict(is_gpu=True, maxit=100, keep_zeros=True, filters=[
                dict(name='Gabor', sigma=[2, 35], theta=0, noise_level=0.5),  #, frequency=0.3),
            ])],
            ['vsnr', dict(is_gpu=True, maxit=100, keep_zeros=True, filters=[
                dict(name='Gabor', sigma=[2, 35], theta=90, noise_level=0.5),
            ])],
            ['median', dict(radius=[5, 200], filter_mode='subtract', cast_dtype='uint16', sample_footprint=200, elliptical_footprint=True, keep_zeros=True)],
            ['median', dict(radius=[300, 5], filter_mode='subtract', cast_dtype='uint16', sample_footprint=200, elliptical_footprint=True, keep_zeros=True)],
            # ['median', dict(radius=[1, 11], on_binned=9, filter_mode='subtract', cast_dtype='uint16', keep_zeros=True)],
            # ['vsnr', dict(is_gpu=True, maxit=100, keep_zeros=True, filters=[
            #     dict(name='Gabor', sigma=[2, 35], theta=90, noise_level=0.5),  #, frequency=0.3),
            # ])],
            ['clahe', dict(tile_grid_in_pixels=True, tile_grid_size=(63, 63), keep_zeros=True)],
            ['median', dict(radius=2, elliptical_footprint=True, keep_zeros=True)],
        ],
        return_intermediates=True
    )

    from matplotlib import pyplot as plt

    # Example: list of images (replace with your actual images)

    # Number of images
    n = len(intermediates)

    # Create subplots: 1 row, n columns
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))  # adjust size as needed

    # If only one image, axes is not a list, so make it iterable
    if n == 1:
        axes = [axes]

    # Plot each image
    for ax, intermediate in zip(axes, intermediates):
        ax.imshow(intermediate, cmap='gray')  # or remove cmap for color images
        ax.axis('off')  # remove axes ticks

    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.imshow(np.concatenate([img, result], axis=1), cmap='gray')
    plt.show()

