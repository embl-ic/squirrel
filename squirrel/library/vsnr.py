
import numpy as np
import json


def get_vsnr_defaults():

    return dict(
        alpha=1e-2,
        filter='gabor',
        sigma=(1, 30),
        theta=0,
        maxit=100,
        is_gpu=True
    )


def pad_to_even_shape(image):
    original_shape = image.shape

    img_ = np.zeros((np.ceil(np.array(image.shape) / 2) * 2).astype(int), dtype=image.dtype)
    img_[:image.shape[0], :image.shape[1]] = image
    image = img_

    return image, original_shape


def crop_to_bounds(image, even_shape=False):
    from squirrel.library.image import get_bounds

    input_shape = image.shape

    if not even_shape:
        bounds = get_bounds(image)
        return image[bounds], bounds, input_shape

    # Bounds in the format y, x, h, w
    bounds = np.array(get_bounds(image, return_ints=True)).astype(int)

    # Make sure height and width are even
    bounds[2:] = np.array(bounds[0: 2]) + (np.ceil(
        (np.array(bounds[2:]) - np.array(bounds[0: 2])) / 2
    ) * 2).astype(int)

    if (bounds[2:] > np.array(input_shape)).any():
        # Image needs to be padded to compensate
        print(f'image.shape = {image.shape}')
        print(f'bounds = {bounds}')
        image_ = np.zeros(np.max([bounds[2:], image.shape], axis=0), dtype=image.dtype)
        image_[:image.shape[0], :image.shape[1]] = image
        image = image_

    # Translate bounds to the slice object
    bounds = np.s_[bounds[0]: bounds[2], bounds[1]: bounds[3]]

    return image[bounds], bounds, input_shape


def normalize(image):
    dtype = image.dtype
    image = image.astype('float32')
    maximum = image.max()
    image /= maximum
    return image, maximum, dtype


def un_normalize(image, maximum, dtype):
    return (np.clip(image, 0, 1) * maximum).astype(dtype)


def vsnr(
        image,
        is_gpu=True,
        filters=None,
        maxit=20,
        algo='auto'
):

    if filters is None:
        filters = [dict(name='Gabor', noise_level=0.35, sigma=[1, 50], theta=90)]

    # image processing
    if is_gpu:
        from pyvsnr import vsnr2d_cuda
        image = vsnr2d_cuda(image, filters, nite=maxit)
    else:
        from pyvsnr import vsnr2d
        image = vsnr2d(image, filters, maxit=maxit, algo=algo)

    return image


def pad(image, bounds, shape):

    img_ = np.zeros(shape, dtype=image.dtype)
    img_[bounds] = image
    return img_


def vsnr_on_image(
        image,
        filters=None,
        is_gpu=True,
        maxit=20,
        algo='auto',
        verbose=False
):

    # --- Prepare the image for VSNR ---
    if verbose:
        print(f'Preparing image for VSNR ...')

    image, bounds, input_shape = crop_to_bounds(image, even_shape=False)
    if verbose:
        print(f'image.shape = {image.shape}')

    image, image_maximum, image_dtype = normalize(image)

    # --- Run VSNR ---
    if verbose:
        print(f'Running VSNR for image of size {image.shape} with algo={algo} ...')

    image = vsnr(
        image,
        filters=filters if type(filters) is list else [filters],
        is_gpu=is_gpu,
        maxit=maxit,
        algo=algo
    )

    # --- Bring the resulting image back to the input domain ---
    if verbose:
        print(f'Bringing the resulting image back to the input domain ...')

    image = un_normalize(image, image_maximum, image_dtype)

    image = pad(image, bounds, input_shape)

    return image


def change_vsnr_params_from_file(filepath, in_params):

    with open(filepath, mode='r') as f:
        params = json.load(f)

    for k, v in params.items():
        in_params[k] = v

    return in_params


def get_vsnr_params(filepath=None):

    params = get_vsnr_defaults()
    if filepath is not None:
        params = change_vsnr_params_from_file(filepath, params)

    return params

#
# def vsnr_on_image(
#         in_filepath,
#         out_filepath=None,
#         vsnr_param_filepath=None,
#         verbose=False
# ):
#
#     if verbose:
#         print(f'in_filepath = {in_filepath}')
#         print(f'out_filepath = {out_filepath}')
#
#     from tifffile import imread, imwrite
#
#     vsnr_params = get_vsnr_params(vsnr_param_filepath)
#
#     corrected = vsnr_workflow(
#         imread(in_filepath),
#         vsnr_params=vsnr_params,
#         verbose=verbose
#     )
#     imwrite(out_filepath, corrected, compression='zlib')
#
#
# def vsnr_on_stack(
#         in_folder,
#         out_folder,
#         in_pattern='*.tif',
#         vsnr_param_filepath=None,
#         overwrite=False,
#         verbose=False
# ):
#
#     if verbose:
#         print(f'in_folder = {in_folder}')
#         print(f'out_folder = {out_folder}')
#
#     from glob import glob
#     import os
#
#     if not os.path.exists(out_folder):
#         os.mkdir(out_folder)
#
#     im_list = sorted(glob(os.path.join(in_folder, in_pattern)))
#     out_filepaths = [os.path.join(out_folder, os.path.split(fp)[1]) for fp in im_list]
#
#     for idx, im_filepath in enumerate(im_list):
#
#         if not overwrite and os.path.exists(out_filepaths[idx]):
#             continue
#
#         vsnr_on_image(
#             im_filepath, out_filepath=out_filepaths[idx], vsnr_param_filepath=vsnr_param_filepath, verbose=verbose
#         )

