
import numpy as np


def _get_quantiles(array, quantiles=(0.1, 0.9), threshold=(1, 254), dilate_background=0):
    from copy import deepcopy
    this_array = deepcopy(array)
    if dilate_background:
        from vigra.filters import discErosion, gaussianSmoothing
        mask = 1 - discErosion((gaussianSmoothing(this_array, 1.0) > 1).astype('uint8'), dilate_background)
        this_array[mask > 0] = 0
    array_ = this_array[np.logical_and(this_array > threshold[0], this_array < threshold[1])]
    return np.quantile(array_, quantiles[0]), np.quantile(array_, quantiles[1])


def _normalize(pixels, alow, ahigh, qlow, qhigh, keep_zeros=False):
    zeros = None if not keep_zeros else pixels == 0
    max_val = np.iinfo(pixels.dtype).max
    pixels = pixels.astype('float64')
    pixels -= qlow
    pixels /= qhigh - qlow
    pixels *= ahigh - alow
    pixels += alow
    pixels[pixels < 0] = 0
    pixels[pixels > max_val] = max_val
    pixels = np.round(pixels / max_val * 255).astype('uint8')
    if keep_zeros:
        pixels[zeros] = 0
    return pixels


def apply_quantiles(image, quantiles=(0.1, 0.9), anchors=(0.2, 0.8), dilate_background=0, keep_zeros=False):
    threshold = (1, np.iinfo(image.dtype).max - 1)
    qlow, qhigh = _get_quantiles(image, quantiles, threshold=threshold, dilate_background=dilate_background)
    alow, ahigh = np.array(anchors) * np.iinfo(image.dtype).max
    image = _normalize(image, alow, ahigh, qlow, qhigh, keep_zeros=keep_zeros)
    return image


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

    from squirrel.library.data import norm_z_range
    stack_shape = stack.shape
    z_range = norm_z_range(z_range, stack_shape[0])

    if n_workers == 1:

        result_stack = []
        for idx in range(*z_range):
            img = stack[idx]
            result_stack.append(apply_quantiles(img, quantiles, anchors, dilate_background, keep_zeros))

    else:
        print(f'Running with {n_workers} CPUs')
        from multiprocessing import Pool

        with Pool(processes=n_workers) as p:
            tasks = []
            for idx in range(*z_range):
                img = stack[idx]
                tasks.append(p.apply_async(apply_quantiles, (img, quantiles, anchors, dilate_background, keep_zeros)))
            result_stack = [task.get() for task in tasks]

    return np.array(result_stack)


def _adjust_greyscale_in_image(
        img,
        greys_in=None,
        greys_out=None,
        cast_dtype='uint8'
):

    if greys_in is None:
        greys_in = [np.min(img), np.max(img)]
    if greys_out is None:
        greys_out = [np.iinfo(cast_dtype).min, np.iinfo(cast_dtype).max]

    img = img.astype('float32')
    img -= greys_in[0]
    img /= greys_in[1] - greys_in[0]
    img *= greys_out[1] - greys_in[0]
    img += greys_out[0]
    img[img < 0] = 0
    img[img > np.iinfo(cast_dtype).max] = np.iinfo(cast_dtype).max
    return img.astype(cast_dtype)


def adjust_greyscale(
        stack,
        z_range=None,
        greys_in=None,
        greys_out=None,
        cast_dtype='uint8',
        n_workers=1
):

    from squirrel.library.data import norm_z_range
    stack_shape = stack.shape
    z_range = norm_z_range(z_range, stack_shape[0])

    if n_workers == 1:

        result_stack = []
        for idx in range(*z_range):
            img = stack[idx]
            result_stack.append(_adjust_greyscale_in_image(img, greys_in, greys_out, cast_dtype))

    else:
        print(f'Running with {n_workers} CPUs')
        from multiprocessing import Pool

        with Pool(processes=n_workers) as p:
            tasks = []
            for idx in range(*z_range):
                img = stack[idx]
                tasks.append(p.apply_async(_adjust_greyscale_in_image, (img, greys_in, greys_out, cast_dtype)))
            result_stack = [task.get() for task in tasks]

    return np.array(result_stack)


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

    from squirrel.library.data import norm_z_range
    stack_shape = stack.shape
    z_range = norm_z_range(z_range, stack_shape[0])

    if n_workers == 1:

        result_stack = []
        for idx in range(*z_range):
            img = stack[idx]
            result_stack.append(
                clahe_on_image(
                    img, clip_limit, tile_grid_size,
                    cast_dtype, invert_output, gaussian_sigma, auto_mask, background_to_mean, tile_grid_in_pixels
                )
            )

    else:
        print(f'Running with {n_workers} CPUs')
        from multiprocessing import Pool

        with Pool(processes=n_workers) as p:
            tasks = []
            for idx in range(*z_range):
                img = stack[idx]
                tasks.append(
                    p.apply_async(
                        clahe_on_image, (
                            img, clip_limit, tile_grid_size,
                            cast_dtype, invert_output, gaussian_sigma, auto_mask, background_to_mean,
                            tile_grid_in_pixels
                        )
                    )
                )
            result_stack = [task.get() for task in tasks]

    return np.array(result_stack)


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
