import os.path

import numpy as np


def invert_slices_workflow(
        in_path,
        out_path,
        in_pattern='*.tif',
        in_key='data',
        out_key='data',
        z_range=None,
        batch_size=None,
        n_workers=1,
        verbose=False
):

    if verbose:
        print(f'in_path = {in_path}')
        print(f'out_path = {out_path}')
        print(f'pattern = {in_pattern}')
        print(f'in_h5_key = {in_key}')
        print(f'out_h5_key = {out_key}')

    from squirrel.library.io import load_data_handle, write_stack, get_filetype
    # from ..library.data import norm_z_range
    stack_handle, stack_shape = load_data_handle(in_path, in_key, in_pattern)

    from squirrel.library.volume import invert_slices

    if batch_size is None:

        normalized_stack = invert_slices(
            stack_handle,
            z_range=z_range,
            n_workers=n_workers
        )
        write_stack(out_path, normalized_stack, key=out_key)
        return

    assert get_filetype(out_path) == 'dir', 'Batch processing only implemented for tif stack output!'

    for zidx in range(0, stack_shape[0], batch_size):

        normalized_substack = invert_slices(
            stack_handle,
            z_range=[zidx, zidx + batch_size],
            n_workers=n_workers
        )
        from squirrel.library.io import write_tif_stack
        write_tif_stack(normalized_substack, out_path, id_offset=zidx, slice_name='slice_{:05}.tif')


def crop_from_stack_workflow(
        stack_path,
        out_path,
        roi=None,
        key='data',
        pattern='*.tif',
        out_slice_offset=0,
        reslice_sample=False,
        verbose=False
):

    if verbose:
        print(f'stack_path = {stack_path}')
        print(f'roi = {roi}')
        print(f'out_path = {out_path}')
        print(f'key = {key}')
        print(f'pattern = {pattern}')
        print(f'reslice_sample = {reslice_sample}')

    if roi is None and not reslice_sample:
        raise RuntimeError('Either roi must be given or for_reslice set to True')

    from squirrel.library.io import load_data_handle, get_filetype
    from squirrel.library.io import TiffStack

    h, s = load_data_handle(stack_path, key=key, pattern=pattern)

    if not reslice_sample:
        for ridx, r in enumerate(roi[3:]):
            if roi[ridx + 3] == 0: 
                roi[ridx + 3] = s[ridx] - roi[ridx + 3]

        if verbose:
            print(f'roi = {roi}')

        roi = np.s_[roi[0]: roi[0] + roi[3], roi[1]: roi[1] + roi[4], roi[2]: roi[2] + roi[5]]
        if isinstance(h, TiffStack):
            data = h[:][roi]
        else:
            data = h[roi]
    else:
        x_positions = np.array(range(0, h.shape[2] - 64, int((h.shape[2] - 64) / 15)))
        x_positions += int((h.shape[2] - x_positions[-1]) / 2)
        x_positions = tuple(x_positions)
        data = []
        for idx, sl in enumerate(h):
            print(f'{idx + 1} / {h.shape[0]}')
            data.append(sl[:, x_positions])
        data = np.array(data).transpose((2, 1, 0))

    ft_out = get_filetype(out_path)

    if ft_out == 'dir':
        from squirrel.library.io import write_tif_stack
        write_tif_stack(data, out_path, id_offset=out_slice_offset)
        return
    if ft_out == 'h5':
        from squirrel.library.io import write_h5_container
        write_h5_container(out_path, data)
        return
    if ft_out == 'ome_zarr':
        from squirrel.library.ome_zarr import create_ome_zarr, get_ome_zarr_handle, chunk_to_ome_zarr
        create_ome_zarr(
            filepath=out_path,
            shape=data.shape
        )
        h = get_ome_zarr_handle(out_path, mode='a')
        chunk_to_ome_zarr(data, [0, 0, 0], h, key='s0', populate_downsample_layers=True)
        return
    raise ValueError(f'Invalid output type = {ft_out}')


def stack_calculator_workflow(
        stack_paths,
        out_path,
        keys=('data', 'data'),
        patterns=('*.tif', '*.tif'),
        operation='add',
        target_dtype=None,
        n_workers=1,
        verbose=False
):

    if verbose:
        print(f'stack_paths = {stack_paths}')
        print(f'out_path = {out_path}')
        print(f'keys = {keys}')
        print(f'patterns = {patterns}')
        print(f'operation = {operation}')
        print(f'target_dtype = {target_dtype}')
        print(f'n_workers = {n_workers}')

    from squirrel.library.io import load_data_handle, get_filetype

    h0, s0 = load_data_handle(stack_paths[0], key=keys[0], pattern=patterns[0])
    h1, s1 = load_data_handle(stack_paths[1], key=keys[1], pattern=patterns[0])

    assert s0 == s1, 'Both stacks must have equal sizes in all three dimensions!'

    from squirrel.library.volume import stack_calculator
    result = stack_calculator(
        h0, h1, operation=operation,
        target_dtype=target_dtype,
        n_workers=n_workers, verbose=verbose
    )

    ft_out = get_filetype(out_path)

    if ft_out == 'dir':
        from squirrel.library.io import write_tif_stack
        write_tif_stack(result, out_path)
        return
    if ft_out == 'h5':
        from squirrel.library.io import write_h5_container
        write_h5_container(out_path, result)
        return
    raise ValueError(f'Invalid output type = {ft_out}')


def running_average_workflow(
        stack,
        out_filepath,
        key='data',
        pattern='*.tif',
        average_method='mean',
        window_size=None,
        operation=None,
        axis=0,
        z_range=None,
        batch_size=None,
        quiet=False,
        verbose=False
):

    if verbose:
        print(f'stack = {stack}')
        print(f'out_filepath = {out_filepath}')
        print(f'key = {key}')
        print(f'pattern = {pattern}')
        print(f'average_method = {average_method}')
        print(f'window_size = {window_size}')
        print(f'operation = {operation}')
        print(f'axis = {axis}')
        print(f'z_range = {z_range}')
        print(f'batch_size = {batch_size}')
        print(f'quiet = {quiet}')

    from squirrel.library.io import load_data_handle, write_stack
    from squirrel.library.data import norm_z_range
    from squirrel.library.volume import running_volume_average

    stack, stack_size = load_data_handle(stack, key=key, pattern=pattern)

    z_range = norm_z_range(z_range, stack_size[0])
    if batch_size is None:
        batch_size = stack_size[0]

    result = []

    for idx in range(*z_range, batch_size):
        if not quiet:
            print(f'idx = {idx} / {z_range[1]}')

        result.append(
            running_volume_average(
                stack[idx: idx + batch_size],
                average_method=average_method,
                window_size=window_size,
                operation=operation,
                axis=axis,
                verbose=verbose
            )
        )

    result = np.concatenate(result, axis=0)
    write_stack(out_filepath, result, key)


def axis_median_filter_workflow(
        stack,
        out_path,
        key='data',
        pattern='*.tif',
        median_radius=2,
        axis=0,
        operation=None,
        z_range=None,
        batch_size=None,
        n_workers=1,
        quiet=False,
        verbose=False
):

    if verbose:
        print(f'stack = {stack}')
        print(f'out_path = {out_path}')
        print(f'key = {key}')
        print(f'pattern = {pattern}')
        print(f'median_radius = {median_radius}')
        print(f'axis = {axis}')
        print(f'operation = {operation}')
        print(f'z_range = {z_range}')
        print(f'batch_size = {batch_size}')
        print(f'n_workers = {n_workers}')
        print(f'quiet = {quiet}')

    from squirrel.library.io import load_data_handle, write_stack
    from squirrel.library.data import norm_z_range
    from squirrel.library.volume import axis_median_filter

    stack, stack_size = load_data_handle(stack, key=key, pattern=pattern)

    z_range = norm_z_range(z_range, stack_size[0])
    if batch_size is None:
        batch_size = stack_size[0]

    result = []

    for idx in range(*z_range, batch_size):
        if not quiet:
            print(f'idx = {idx} / {z_range[1]}')

        result.append(
            axis_median_filter(
                stack[idx: idx + batch_size],
                median_radius=median_radius,
                axis=axis,
                operation=operation,
                n_workers=n_workers,
                verbose=verbose
            )
        )

    result = np.concatenate(result, axis=0)
    write_stack(out_path, result, key)


def _get_label_list(data_h, z_range):

    ids = []
    for idy in range(0, data_h.shape[1], data_h.chunks[1]):
        for idx in range(0, data_h.shape[2], data_h.chunks[2]):
            print(f'loading chunk idx, idy, idz: {idx}, {idy}, {z_range[0]}')
            try:
                data = data_h[
                       z_range[0]: z_range[1],
                       idy: idy + data_h.chunks[1],
                       idx: idx + data_h.chunks[2],
                ]
            except TypeError:
                # For tiff slices it has to be done like this
                data = data_h[
                    z_range[0]: z_range[1]
                ][
                    idy: idy + data_h.chunks[1],
                    idx: idx + data_h.chunks[2],
                ]
            ids.append(np.unique(data))

    return np.unique(np.concatenate(ids))


def get_label_list_workflow(
        input_path,
        key=None,
        pattern=None,
        out_json=None,
        z_batch_size=1,
        n_workers=1,
        quiet=False,
        verbose=False
):

    if verbose:
        print(f'input_path = {input_path}')
        print(f'key = {key}')
        print(f'pattern = {pattern}')
        print(f'out_json = {out_json}')
        print(f'z_batch_size = {z_batch_size}')
        print(f'n_workers = {n_workers}')

    from squirrel.library.io import load_data_handle
    h, shape = load_data_handle(input_path, key, pattern)
    print(f'h.chunks = {h.chunks}')

    label_lists = []

    if n_workers == 1:
        for idx in range(0, shape[0], z_batch_size):
            z_range = [idx, min(idx + z_batch_size, shape[0])]
            label_lists.append(_get_label_list(h, z_range))

    else:

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as tpe:
            tasks = [
                tpe.submit(_get_label_list, h, [idx, min(idx + z_batch_size, shape[0])])
                for idx in range(0, shape[0], z_batch_size)
            ]
            label_lists = [task.result() for task in tasks]

    label_list = np.unique(np.concatenate(label_lists))

    if out_json is not None:
        # Write label-list to file
        import json
        with open(out_json, mode='w') as f:
            json.dump(label_list.tolist(), f, indent=2)

    if out_json is None or verbose and not quiet:
        print(f'label_list = {[x for x in label_list]}')

    return label_list


def tif_nearest_scaling_workflow(
        input_dirpath,
        output_dirpath,
        pattern='*.tif',
        scale_factors=[1., 1., 1.],
        n_workers=1,
        verbose=False
):

    if not os.path.exists(output_dirpath):
        os.mkdir(output_dirpath)

    # Load the input tif stack (just as a handle, not the data)
    from squirrel.library.io import TiffStack
    ts = TiffStack(input_dirpath, pattern=pattern)
    input_shape = ts.shape

    if verbose:
        print(f'input_shape = {input_shape}')
        print(f'scale_factors = {scale_factors}')

    # Determine mapping of input to output z-slices
    from squirrel.library.scaling import create_nearest_position_mapping
    nearest_z_position_mapping = create_nearest_position_mapping(input_shape[0], scale_factors[0])
    if verbose:
        print(f'nearest_z_position_mapping = {nearest_z_position_mapping}')

    from shutil import copy2
    from squirrel.library.io import write_tif_slice
    from squirrel.library.scaling import scale_image_nearest
    # Now iterate the relevant z-slices, scale them and write them to the target dataset
    for out_idz, in_idz in nearest_z_position_mapping.items():
        if verbose:
            print(f'out_idz = {out_idz}; in_idz = {in_idz}')

        if scale_factors[1] == 1. and scale_factors[2] == 1.:
            # Just copy the slices
            src_filepath = ts.get_filepaths()[in_idz]
            tgt_filepath = os.path.join(
                output_dirpath, 'slice_{:05d}.tif'.format(out_idz)
            )
            copy2(src_filepath, tgt_filepath)

        else:
            # Do the respective x- and y-scaling for the slices
            in_img = ts[in_idz]
            # out_shape = np.array(input_shape)[1:] * np.array(scale_factors)[1:]
            out_img = scale_image_nearest(in_img, scale_factors[1:])
            if verbose:
                print(f'out_img.shape = {out_img.shape}')
            write_tif_slice(out_img, output_dirpath, 'slice_{:05d}.tif'.format(out_idz))


def estimate_crop_xy_workflow(
        input_path,
        key=None,
        pattern=None,
        padding=64,
        out_image='max_projection.png',
        number_of_samples=16,
        verbose=False
):

    if verbose:
        print(f'input_path = {input_path}')
        print(f'key = {key}')
        print(f'pattern = {pattern}')
        print(f'padding = {padding}')
        print(f'out_image = {out_image}')
        print(f'number_of_samples = {number_of_samples}')

    # Load the relevant slices
    from squirrel.library.io import load_data_handle
    h, shape = load_data_handle(input_path, key, pattern)

    slice_ids = np.array(range(number_of_samples)) / (number_of_samples - 1) * (shape[0] - 1)
    slice_ids = np.round(slice_ids).astype(int)
    if verbose:
        print(f'shape[0] = {shape[0]}')
        print(f'slice_ids = {slice_ids}')

    slices = h[tuple(slice_ids)]

    if verbose:
        print(f'slices.shape = {slices.shape}')

    # Compute the max projection
    max_projection = slices.max(axis=0)

    if verbose:
        print(f'max_projection.shape = {max_projection.shape}')

    # Determine the bounds and add the padding
    from squirrel.library.image import get_bounds
    bounds = get_bounds(max_projection, return_ints=True)
    bounds_start = np.array(bounds[:2])
    bounds_stop = np.array(bounds[2:])
    bounds_start -= padding
    bounds_start[bounds_start < 0] = 0
    bounds_stop += padding
    shape_yx = np.array(shape[1:])
    bounds_stop[bounds_stop > shape_yx] = shape_yx[bounds_stop > shape_yx]
    bounds = np.concatenate((bounds_start, bounds_stop), axis=0).tolist()

    if verbose:
        from matplotlib import pyplot as plt
        plt.imshow(max_projection)
        plt.show()

    # Save the maximum projection (with bounds?)
    from PIL import Image, ImageDraw

    def _draw_rectangle_on_image(
            image_array, rectangle, color=(255, 0, 0), alpha=32, width=3, output_path='output.png'
    ):
        """
        Draws a semi-transparent rectangle on a grayscale image and saves it as a PNG.

        Parameters:
            image_array (np.ndarray): 2D grayscale image array.
            rectangle (list or tuple): [y_start, x_start, y_stop, x_stop].
            color (tuple): RGB color of the rectangle.
            alpha (int): Transparency (0–255), 128 = 50% transparent.
            width (int): Border thickness in pixels.
            output_path (str): Path to save the output PNG.
        """
        # Convert grayscale image to RGB then to RGBA
        base_img = Image.fromarray(image_array.astype(np.uint8), mode='L').convert('RGBA')

        # Create a transparent overlay
        overlay = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        y0, x0, y1, x1 = rectangle
        rgba_color = (*color, alpha)
        draw.rectangle((x0, y0, x1-1, y1-1), outline=rgba_color, width=width)
        draw.rectangle((x0, y0, x1-1, y1-1), outline=color, width=int(max(image_array.shape) / 500))

        # Composite the overlay onto the base image
        result = Image.alpha_composite(base_img, overlay)
        result.save(output_path)
        print(f"Saved maximum projection to: {output_path}")

    if out_image is not None:
        # if os.path.exists(os.path.split(out_image)[0]):
        #     out_filepath = out_image
        # else:
        #     out_filepath = os.path.join(os.path.split(input_path)[0], out_image)
        out_filepath = out_image
        _draw_rectangle_on_image(
            max_projection,
            bounds,
            color=(255, 128, 0),
            output_path=out_filepath,
            width=200
        )

    # Print bounds
    # Convert to YXHW
    bounds[2:] = [bounds[2] - bounds[0], bounds[3] - bounds[1]]
    # Convert to XYWH order
    bounds = bounds[:2][::-1] + bounds[2:][::-1]
    print('____________________________________')
    print('BOUNDS (X, Y, W, H):')
    print(f"[{', '.join([str(int(x)) for x in bounds])}]")
    print(f"[{','.join([str(int(x)) for x in bounds])}]")
    print(' '.join([str(int(x)) for x in bounds]))
    print('____________________________________')
    return [int(x) for x in bounds]


def filter_2d_workflow(
        input_path,
        out_path,
        key=None,
        pattern=None,
        out_key=None,
        filters=None,
        batch_size=None,
        n_workers=1,
        verbose=False
):
    if verbose:
        print(f'input_path = {input_path}')
        print(f'out_path = {out_path}')
        print(f'key = {key}')
        print(f'pattern = {pattern}')
        print(f'filters = {filters}')
        print(f'batch_size = {batch_size}')
        print(f'n_workers = {n_workers}')

    from squirrel.library.filters import ImageFilter
    from squirrel.library.io import load_data_handle, write_stack, get_filetype

    stack_handle, stack_shape = load_data_handle(input_path, key, pattern)

    if batch_size is None:

        imf = ImageFilter(stack_handle[:])
        filtered_stack = imf.get_filtered_stack(filters, n_workers=n_workers)
        write_stack(out_path, filtered_stack, key=out_key)
        return

    assert get_filetype(out_path) == 'dir', 'Batched processing only implemented for tif stack output!'

    from squirrel.library.io import write_tif_stack
    for zidx in range(0, stack_shape[0], batch_size):

        imf = ImageFilter(stack_handle[zidx: zidx + batch_size])
        filtered_substack = imf.get_filtered_stack(filters, n_workers=n_workers)
        write_tif_stack(filtered_substack, out_path, id_offset=zidx, slice_name='slice_{:05}.tif')


if __name__ == '__main__':

    filter_2d_workflow(
        '/media/julian/Data/projects/woller/problem-area1/problem_area',
        '/media/julian/Data/tmp/test_stack_filter',
        filters=[
            ['gaussian', dict(sigma=5.0)],
            ['gaussian_gradient_magnitude', dict(sigma=1.0)]
        ],
        batch_size=8, n_workers=8
    )

    # crop_from_stack_workflow(
    #     '/media/julian/Data/projects/woller/problem-area1/problem_area',
    #     '/media/julian/Data/tmp/crop_from_stack_test.h5',
    #     roi=None,
    #     pattern='*.tif',
    #     reslice_sample=True,
    #     verbose=False
    # )

    # estimate_crop_xy_workflow(
    #     '/media/julian/Data/projects/woller/problem-area1/problem_area',
    #     number_of_samples=4,
    #     padding=64,
    #     verbose=True
    # )

    # stack_calculator_workflow(
    #     ('/media/julian/Data/projects/walter/cryo_fib_preprocessing/2024-03-21_2h/InLensCombined',
    #      '/media/julian/Data/projects/walter/cryo_fib_preprocessing/2024-03-21_2h/InLensCombined'),
    #     '/tmp/test_stack_calculator',
    #     operation='average',
    #     n_workers=16,
    #     verbose=True
    # )

    # running_average_workflow(
    #     '/media/julian/Data/projects/ionescu/cryofib-achromarium-segmentation/2022-02-23_giant-bacteria/segmentations/chunks/chunk_0000_2048_0000.h5',
    #     '/media/julian/Data/tmp/test-running-volume-avg.h5',
    #     key='em',
    #     average_method='median',
    #     window_size=200,
    #     operation='difference-clip',
    #     axis=1,
    #     verbose=True
    # )

    # axis_median_filter_workflow(
    #     '/media/julian/Data/projects/ionescu/cryofib-achromarium-segmentation/2022-02-23_giant-bacteria/segmentations/chunks/chunk_0000_2048_0000.h5',
    #     '/media/julian/Data/tmp/test-axis_median.h5',
    #     key='em',
    #     median_radius=200,
    #     axis=1,
    #     operation='difference-clip',
    #     verbose=True
    # )
    #
    # axis_median_filter_workflow(
    #     '/media/julian/Data/tmp/test-axis_median.h5',
    #     '/media/julian/Data/tmp/test-axis_median-ax2.h5',
    #     key='em',
    #     median_radius=200,
    #     axis=2,
    #     operation='difference-clip',
    #     verbose=True
    # )
