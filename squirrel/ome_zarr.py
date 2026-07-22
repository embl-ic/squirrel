
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
    parser.add_argument('--downsample_type', type=str, default='Average',
                        help='How the sampling to compute the resolution pyramid is performed; Default="Average"; possible values: ("Average", "Sample")')
    parser.add_argument('--downsample_factors', type=int, nargs='+', default=(2, 2, 2),
                        help='Specifies the downsample factors for the resolution pyramid; Default=(2, 2, 2)')
    parser.add_argument('--chunk_size', type=int, nargs=3, default=(1, 256, 256),
                        metavar=('Z', 'Y', 'X'),
                        help='Chunk size of the ome.zarr dataset; Default=(1, 256, 256)')
    parser.add_argument('--dtype', type=str, default='uint8',
                        help='Data type of the ome.zarr dataset')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    filepath = args.filepath
    shape = args.shape
    resolution = args.resolution
    unit = args.unit
    downsample_type = args.downsample_type
    downsample_factors = args.downsample_factors
    chunk_size = args.chunk_size
    dtype = args.dtype
    verbose = args.verbose

    from squirrel.workflows.ome_zarr import create_ome_zarr_workflow

    create_ome_zarr_workflow(
        filepath,
        shape,
        resolution=resolution,
        unit=unit,
        downsample_type=downsample_type,
        downsample_factors=downsample_factors,
        chunk_size=chunk_size,
        dtype=dtype,
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
    parser.add_argument('-pdl', '--populate_downsample_layers', action='store_true',
                        help='If set, the downsample layers will be computed for the area of the newly added data')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    data_filepath = args.data_filepath
    oz_filepath = args.oz_filepath
    position = args.position
    data_key = args.data_key
    populate_downsample_layers = args.populate_downsample_layers
    verbose = args.verbose

    from squirrel.workflows.ome_zarr import data_to_ome_zarr_workflow

    data_to_ome_zarr_workflow(
        data_filepath,
        oz_filepath,
        position,
        data_key=data_key,
        populate_downsample_layers=populate_downsample_layers,                
        verbose=verbose
    ) 
