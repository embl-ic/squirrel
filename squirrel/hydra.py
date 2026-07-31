
def get_acquisition_transforms():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Determines transformations based on tiff metadata',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('dirpath', type=str,
                        help='Directory containing the tif slices')
    parser.add_argument('out_filepath', type=str,
                        help='Output json which will contain the transformations.')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    dirpath = args.dirpath
    out_filepath = args.out_filepath
    verbose = args.verbose

    from squirrel.workflows.hydra import get_acquisition_transforms_workflow

    get_acquisition_transforms_workflow(
        dirpath,
        out_filepath=out_filepath,
        verbose=verbose
    )
