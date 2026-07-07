
import numpy as np


def _normalize_chunk_size(chunk_size, n_levels):
    chunk_size = np.array(chunk_size).tolist()
    if isinstance(chunk_size[0], int):
        return np.array([chunk_size] * n_levels)
    if isinstance(chunk_size[0], list):
        assert len(chunk_size) == n_levels
        return np.array(chunk_size)
    raise ValueError(f'Invalid chunk size: {chunk_size}')


def create_ome_zarr(
        filepath,
        shape,
        resolution=(1., 1., 1.),
        unit='pixel',
        downsample_type='Average',  # One of ['Average', 'Sample']
        downsample_factors=(2, 2, 2),
        chunk_size=(1, 256, 256),
        dtype='uint8',
        name=None,
        verbose=False
):

    if verbose:
        print(f'cunk_size = {chunk_size}')
    chunk_size = _normalize_chunk_size(chunk_size, len(downsample_factors) + 1)
    if verbose: 
        print(f'chunk_size = {chunk_size}')

    from zarr import open as zarr_open
    from numcodecs import Blosc
    handle = zarr_open(filepath, mode='w')
    handle.create_dataset(
        's0',
        shape=shape,
        compressor=Blosc(cname='zstd', clevel=9),
        chunks=chunk_size[0],
        dtype=dtype,
        dimension_separator='/'
    )

    if name == None:
        import os
        name = str.replace(os.path.split(filepath)[1], '.ome.zarr', '')

    def dataset(path, scale):
        return dict(
            path=path,
            coordinateTransformations=[
                dict(
                    type='scale',
                    scale=scale
                )
            ]
        )

    def axes():
        return [
            dict(
                name=axis,
                type='space',
                unit=unit
            )
            for axis in 'zyx'
        ]

    handle.attrs.update(
        dict(
            multiscales=[
                dict(
                    axes=axes(),
                    datasets=[dataset('s0', list(resolution))],
                    name=name,
                    type=downsample_type,
                    version="0.4"
                )
            ]
        )
    )

    scale = 1
    s_idx = 0
    for downsample_factor in downsample_factors:

        scale *= downsample_factor
        s_idx += 1

        handle.create_dataset(
            f's{s_idx}',
            shape=(np.array(shape) / scale).astype(int).tolist(),
            compression='gzip',
            chunks=chunk_size[s_idx],
            dtype=dtype,
            dimension_separator='/'
        )

        attrs = handle.attrs
        attrs['multiscales'][0]['datasets'].append(
            dataset(f's{s_idx}', list(np.array(resolution) * scale))
        )

    handle.attrs.update(attrs)

    return handle


def get_ome_zarr_handle(
        filepath,
        key=None,
        mode='r'
):
    from zarr import open as zarr_open
    if key is not None:
        return zarr_open(filepath, mode=mode)[key]
    return zarr_open(filepath, mode=mode)


def get_downsample_order(ome_zarr_handle):

    ds_type = ome_zarr_handle.attrs['multiscales'][0]['type']

    if ds_type == 'Average':
        return 0
    if ds_type == 'Sample':
        return 1
    raise ValueError(f'Invalid down-sample type: {ds_type}')


def downsample_ome_zarr_chunk(
        chunk_position,
        chunk_shape,
        ome_zarr_handle,
        verbose=False
):

    from ..library.affine_matrices import AffineMatrix
    from ..library.transformation import setup_scale_matrix, apply_affine_transform

    downsample_factors = [1] + get_downsample_factors(ome_zarr_handle)

    print(f'chunk_position = {chunk_position}')
    print(f'chunk_shape = {chunk_shape}')
    print(f'downsample_factors = {downsample_factors}')

    chunk_position = np.array(chunk_position)
    chunk_shape = np.array(chunk_shape)
    prev_v = None

    for idx, (k, v) in enumerate(ome_zarr_handle.items()):
        chunks = np.array(v.chunks)
        shape = np.array(v.shape)
        prev_chunk_position = chunk_position.copy()
        prev_chunk_shape = chunk_shape.copy()
        chunk_position = (chunk_position / downsample_factors[idx]).astype(int)
        chunk_shape = (chunk_shape / downsample_factors[idx]).astype(int)

        # Some assertions to make sure the chunks boundaries match the requested location
        for dim in range(v.ndim):
            assert chunk_shape[dim] / chunks[dim] == int(chunk_shape[dim] / chunks[dim]) or chunk_position[dim] + chunk_shape[dim] == shape[dim], \
                f"Given chunk size and ome_zarr chunks boundaries don't match for {k} in dimension {dim}; " \
                f"{chunk_shape[dim] / chunks[dim]} != {int(chunk_shape[dim] / chunks[dim])} and " \
                f"{chunk_position[dim] + chunk_shape[dim]} != {shape[dim]}"
            assert chunk_position[dim] / chunks[dim] == int(chunk_position[dim] / chunks[dim]), \
                f"Given chunk position and ome_zarr chunks boundaries don't match for {k} in dimension {dim}; " \
                f"{chunk_position[dim] / chunks[dim]} != {int(chunk_position[dim] / chunks[dim])}"

        if idx > 0:

            downsample_factor = downsample_factors[idx]
            downsample_order = get_downsample_order(ome_zarr_handle)

            transform_matrix = AffineMatrix(
                parameters=setup_scale_matrix([downsample_factor] * 3, ndim=3).flatten()
            )

            source_data = prev_v[prev_chunk_position[0]: prev_chunk_position[0] + prev_chunk_shape[0],
                                 prev_chunk_position[1]: prev_chunk_position[1] + prev_chunk_shape[1],
                                 prev_chunk_position[2]: prev_chunk_position[2] + prev_chunk_shape[2]]

            target_data = apply_affine_transform(
                source_data,
                transform_matrix,
                order=downsample_order,
                scale_canvas=True,
                no_offset_to_center=True,
                verbose=verbose
            )

            v[chunk_position[0]: chunk_position[0] + chunk_shape[0],
              chunk_position[1]: chunk_position[1] + chunk_shape[1],
              chunk_position[2]: chunk_position[2] + chunk_shape[2]] = target_data

        prev_v = v


def chunk_to_ome_zarr(
        chunk_data,
        chunk_position,
        ome_zarr_handle,
        key='s0',
        populate_downsample_layers=False,
        verbose=False
):

    sl = np.s_[
        chunk_position[0]: chunk_position[0] + chunk_data.shape[0],
        chunk_position[1]: chunk_position[1] + chunk_data.shape[1],
        chunk_position[2]: chunk_position[2] + chunk_data.shape[2]
    ]
    print(f'sl = {sl}')
    ome_zarr_handle[key][sl] = chunk_data

    if populate_downsample_layers:
        assert key == 's0'
        downsample_ome_zarr_chunk(
            chunk_position,
            chunk_data.shape,
            ome_zarr_handle,
            verbose=verbose
        )


# NOTE: Deprecated
# def slice_to_ome_zarr(
#         stack_path,
#         slice_idx,
#         ome_zarr_handle,
#         stack_key='data',
#         stack_pattern='*.tif',
#         save_bounds=False,
#         verbose=False
# ):
#     print(f'slice_idx = {slice_idx}')
#     if verbose:
#         print(f'stack_path = {stack_path}')
#         print(f'ome_zarr_handle = {ome_zarr_handle}')
#         print(f'save_bounds = {save_bounds}')
#
#     from .io import load_data_from_handle_stack, load_data_handle
#
#     data_handle, shape_h = load_data_handle(stack_path, key=stack_key, pattern=stack_pattern)
#     slice_data, _ = load_data_from_handle_stack(data_handle, slice_idx)
#     if verbose:
#         print(f'loaded slice data...')
#
#     if save_bounds:
#         # TODO this
#         pass
#
#     ome_zarr_handle[slice_idx, :] = slice_data
#     if verbose:
#         print(f'slice data written')
#
#
# def process_slice_to_ome_zarr(
#         stack_path,
#         z_range,
#         ome_zarr_filepath,
#         stack_key='data',
#         stack_pattern='*.tif',
#         save_bounds=False,
#         n_threads=1,
#         verbose=False
# ):
#
#     ome_zarr_h = get_ome_zarr_handle(ome_zarr_filepath, 's0', 'a')
#
#     if n_threads == 1:
#
#         for idx in range(*z_range):
#             slice_to_ome_zarr(
#                 stack_path,
#                 idx,
#                 ome_zarr_h,
#                 stack_key=stack_key,
#                 stack_pattern=stack_pattern,
#                 save_bounds=save_bounds,
#                 verbose=verbose
#             )
#
#     else:
#
#         from multiprocessing import Pool
#         with Pool(processes=n_threads) as p:
#             tasks = [
#                 p.apply_async(slice_to_ome_zarr, (
#                     stack_path,
#                     idx,
#                     ome_zarr_h,
#                     stack_key,
#                     stack_pattern,
#                     save_bounds,
#                     verbose
#                 ))
#                 for idx in range(*z_range)
#             ]
#             [task.get() for task in tasks]
#
#
# def slice_of_downsampling_layer(
#         source_ome_zarr_handle,
#         target_ome_zarr_handle,
#         # source_slice_idx,
#         target_slice_idx,
#         downsample_factor=2,
#         downsample_order=1,
#         verbose=False
# ):
#
#     print(f'target_slice_idx = {target_slice_idx}')
#     if verbose:
#         print(f'source_ome_zarr_handle = {source_ome_zarr_handle}')
#         print(f'target_ome_zarr_handle = {target_ome_zarr_handle}')
#         print(f'downsample_factor = {downsample_factor}')
#
#     from .transformation import apply_affine_transform, setup_scale_matrix, validate_and_reshape_matrix
#
#     source_slice_idx = target_slice_idx * downsample_factor
#     source_data = source_ome_zarr_handle[source_slice_idx: source_slice_idx + downsample_factor]
#     if downsample_order == 0:
#         source_data = source_data[int(downsample_factor / 2) - 1, :]
#     elif downsample_order == 1:
#         source_data = np.mean(source_data, axis=0)
#     else:
#         raise NotImplementedError(f'Downsample order = {downsample_order} is not implemented!')
#     transform_matrix = validate_and_reshape_matrix(
#         setup_scale_matrix([downsample_factor] * 2, ndim=2),
#         ndim=2
#     )
#     target_data = apply_affine_transform(
#         source_data,
#         transform_matrix,
#         order=downsample_order,
#         scale_canvas=True,
#         no_offset_to_center=True,
#         verbose=verbose
#     )
#     # # For a reason that I currently don't understand this does not work
#     # #   Regardless of downsample_order is always produces as if it was downsample_order=0
#     # transform_matrix = validate_and_reshape_matrix(
#     #     setup_scale_matrix([downsample_factor] * 3, ndim=3),
#     #     ndim=3
#     # )
#     # target_data = apply_affine_transform(
#     #     source_data,
#     #     transform_matrix,
#     #     order=downsample_order,
#     #     scale_canvas=True,
#     #     no_offset_to_center=True,
#     #     verbose=verbose
#     # )
#
#     # this_slice_idx = int(source_slice_idx / downsample_factor)
#     this_slice_idx = target_slice_idx
#     target_data = target_data.squeeze()
#     try:
#         target_ome_zarr_handle[this_slice_idx, :] = target_data
#     except ValueError:
#         # This happens if, due to downscaling, the target slice is one pixel larger than the ome-zarr dataset
#         this_shape = target_ome_zarr_handle[this_slice_idx].shape
#         target_ome_zarr_handle[this_slice_idx, :] = target_data[:this_shape[0], :this_shape[1]]
#
#
# def compute_downsampling_layer(
#         ome_zarr_filepath,
#         z_range,
#         source_layer,
#         target_layer,
#         downsample_factor=2,
#         downsample_order=1,
#         n_threads=1,
#         verbose=False
# ):
#
#     source_h = get_ome_zarr_handle(ome_zarr_filepath, source_layer, 'r')
#     target_h = get_ome_zarr_handle(ome_zarr_filepath, target_layer, 'a')
#
#     if n_threads == 1:
#
#         for idx in range(*z_range, downsample_factor):
#             slice_of_downsampling_layer(
#                 source_h,
#                 target_h,
#                 idx,
#                 downsample_factor,
#                 downsample_order=downsample_order,
#                 verbose=verbose
#             )
#     else:
#
#         from multiprocessing import Pool
#         with Pool(processes=n_threads) as p:
#             tasks = [
#                 p.apply_async(slice_of_downsampling_layer, (
#                     source_h,
#                     target_h,
#                     idx,
#                     downsample_factor,
#                     downsample_order,
#                     verbose
#                 ))
#                 for idx in range(*z_range)
#             ]
#             [task.get() for task in tasks]


def get_scale_of_downsample_level(handle, downsample_level):

    datasets = handle.attrs['multiscales'][0]['datasets']
    this_dataset = datasets[downsample_level]
    this_path = this_dataset['path']
    assert this_path == f's{downsample_level}', \
        f'Invalid path to downsample level combination: {this_path} != s{downsample_level}'

    return this_dataset['coordinateTransformations'][0]['scale']


def get_unit_of_dataset(handle):
    units = [x['unit'] for x in handle.attrs['multiscales'][0]['axes']]
    assert units[0] == units[1] == units[2]
    return units[0]


def get_downsample_factors(handle):
    def _get_res_from_dataset(ds):
        return np.array(ds['coordinateTransformations'][0]['scale'])
    datasets = handle.attrs['multiscales'][0]['datasets']

    resolutions = [_get_res_from_dataset(downsampled_dataset) for downsampled_dataset in datasets]

    scales = []
    for idx in range(len(resolutions) - 1):
        resolution = resolutions[idx + 1]
        ref_resolution = resolutions[idx]
        this_scale = (resolution / ref_resolution).astype(int).tolist()
        assert this_scale[0] == this_scale[1] == this_scale[2]
        this_scale = this_scale[0]
        scales.append(this_scale)

    return scales


if __name__ == '__main__':
    
    ozh = get_ome_zarr_handle('/media/julian/Data/tmp/amst2_test_4t_3/pre-align-2.ome.zarr', mode='a')
    shp = ozh.s0.shape
    downsample_ome_zarr_chunk([16, 0, 0], [16, shp[1], shp[2]], ozh)
