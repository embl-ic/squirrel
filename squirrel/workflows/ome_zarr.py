
import os


def create_ome_zarr_workflow(
        filepath,
        shape,
        resolution=(1., 1., 1.),
        unit='pixel',
        downsample_method='Average',  # One of ['Average', 'Sample']
        downsample_factors=(2, 2, 2),
        chunk_size=(1, 256, 256),
        dtype='uint8',
        ome_version='0.4',
        zarr_format=2,
        verbose=False
):

    from squirrel.library.ome_zarr import OMEZarrStore
    return OMEZarrStore.create(
        filepath, 
        shape=shape,
        dtype=dtype,
        chunks=chunk_size,
        downsample_factors=downsample_factors,
        resolution=resolution,
        unit=unit,
        downsample_method=downsample_method,
        ome_version=ome_version,
        zarr_format=zarr_format,
        overwrite=False
    )


def data_to_ome_zarr_workflow(
        data,
        oz_filepath,
        position,
        data_key=None,
        data_pattern=None,
        update_pyramid_mode='data',
        require_empty=False,
        check_alignment=False,
        check_pyramid_alignment=False,
        verbose=False
):

    if update_pyramid_mode not in ['data', 'full', 'none']:
        raise ValueError('Invalid update_pyramid_mode! Possible values: ["data", "full", "none"]')

    from squirrel.library.io import load_data_handle

    if type(data) == str:
        h, _ = load_data_handle(data, key=data_key, pattern=data_pattern)
        data = h[:]

    from squirrel.library.ome_zarr import OMEZarrStore
    store = OMEZarrStore(path=oz_filepath, mode='a')

    store.write(
        0, position, 
        data=data,
        update_pyramid=(update_pyramid_mode == 'data'),
        require_empty=require_empty,
        check_alignment=check_alignment,
        check_pyramid_alignment=check_pyramid_alignment
    )

    if update_pyramid_mode == 'full':
        store.rebuild_pyramid(n_threads=os.cpu_count())


if __name__ == '__main__':

    import numpy as np

    oz_fp = '/media/julian/Data/tmp/create_ome_zarr_wf_test.ome.zarr'

    create_ome_zarr_workflow(
        oz_fp,
        shape=(256, 256, 256),
        chunk_size=(64, 64, 64),
        dtype='uint8'
    )

    data_to_ome_zarr_workflow(
        np.ones((128, 128, 128), dtype='uint8') * 255,
        oz_fp,
        (32, 32, 32),
        data_key= 'data',
        oz_key='s0',
        populate_downsample_layers=False,
        verbose=True
    )
