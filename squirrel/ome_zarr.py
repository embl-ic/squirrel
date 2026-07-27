
def create_ome_zarr():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Creates an empty ome-zarr dataset',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('filepath', type=str,
                        help='Path of the ome-zarr (*.ome.zarr)')
    parser.add_argument('shape', type=int, nargs=3,
                        help='Shape of the dataset (Z Y X)')
    parser.add_argument('--resolution', type=float, nargs=3, default=(1., 1., 1.),
                        metavar=('Z', 'Y', 'X'),
                        help='Resolution; default=(1., 1., 1.)')
    parser.add_argument('--unit', type=str, default='pixel',
                        help='Unit of the resolution entry; default="pixel"')
    parser.add_argument('--downsample_method', type=str, default='Average',
                        help='How the sampling to compute the resolution pyramid is performed; Default="Average"; possible values: ("Average", "Sample")')
    parser.add_argument('--downsample_factors', type=int, nargs='+', default=(2, 2, 2),
                        help='Specifies the downsample factors for the resolution pyramid; Default=(2, 2, 2)')
    parser.add_argument('--chunk_size', type=int, nargs=3, default=(1, 256, 256),
                        metavar=('Z', 'Y', 'X'),
                        help='Chunk size of the ome.zarr dataset; Default=(1, 256, 256)')
    parser.add_argument('--dtype', type=str, default='uint8',
                        help='Data type of the ome.zarr dataset')
    parser.add_argument('--ome_version', type=str, default='0.4',
                        help='Ome version')
    parser.add_argument('--zarr_format', type=int, default=2,
                        help='Zarr format version')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    filepath = args.filepath
    shape = args.shape
    resolution = args.resolution
    unit = args.unit
    downsample_method = args.downsample_method
    downsample_factors = args.downsample_factors
    chunk_size = args.chunk_size
    dtype = args.dtype
    ome_version = args.ome_version
    zarr_format = args.zarr_format
    verbose = args.verbose

    from squirrel.workflows.ome_zarr import create_ome_zarr_workflow

    create_ome_zarr_workflow(
        filepath,
        shape,
        resolution=resolution,
        unit=unit,
        downsample_method=downsample_method,
        downsample_factors=downsample_factors,
        chunk_size=chunk_size,
        dtype=dtype,
        ome_version=ome_version,
        zarr_format=zarr_format,
        verbose=verbose
    )


def data_to_ome_zarr():
   
    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Writes data to an existing ome-zarr dataset',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('data_filepath', type=str,
                        help='Path of the input data (*.h5, *.ome.zarr, tiff stack)')
    parser.add_argument('oz_filepath', type=str,
                        help='Path of the ome-zarr (*.ome.zarr)')
    parser.add_argument('position', type=int, nargs=3, 
                        help='Position in the dataset where the data is placed to (Z Y X)')
    parser.add_argument('-dkey', '--data_key', type=str, default=None,
                        help='In-file path for the input data')
    parser.add_argument('-up', '--update_pyramid_mode', type=str, default='data',
                        help='How to update the resolution pyramid: \n'
                             '    "none": Do not update the resolution pyramid'
                             '    "data" (default): Update for the newly added data; Not parallelized, good for smaller batches of data'
                             '    "full": Updates the full resolution pyramid; Parallelized, good if a large portion of the data was added')
    parser.add_argument('--require_empty', action='store_true', 
                        help='Raise an exception if the target area is not empty')
    parser.add_argument('--check_alignment', action='store_true',
                        help='Raise an exception if the data is not aligned to the storage grid of the target resolution level')
    parser.add_argument('--check_pyramid_alignment', action='store_true',
                        help='Raise an exception if the data is not aligned to the storage grid of any scale level')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    data_filepath = args.data_filepath
    oz_filepath = args.oz_filepath
    position = args.position
    data_key = args.data_key
    update_pyramid_mode = args.update_pyramid_mode
    require_empty = args.require_empty
    check_alignment = args.check_alignment
    check_pyramid_alignment = args.check_pyramid_alignment
    verbose = args.verbose

    from squirrel.workflows.ome_zarr import data_to_ome_zarr_workflow

    data_to_ome_zarr_workflow(
        data_filepath,
        oz_filepath,
        position,
        data_key=data_key,
        update_pyramid_mode=update_pyramid_mode,
        require_empty=require_empty,
        check_alignment=check_alignment,
        check_pyramid_alignment=check_pyramid_alignment,                
        verbose=verbose
    ) 
