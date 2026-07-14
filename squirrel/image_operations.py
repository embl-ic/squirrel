
def filter_2d():
    # ----------------------------------------------------
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description='Performs a 2D filter on each slice of an image stack',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_filepath', type=str,
                        help='Filepath of the input image (*.tif)')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the result image (*.tif)')
    parser.add_argument('-f', '--filters', type=json.loads, nargs='+', default=None,
                        help='Definition for the image filters, use valid json format to define. \n'
                             'The general structure is\n'
                             '  --filters \'["filter1", {"arg1": value1, ...}], ["filter2", {"arg2": value2, ...}, ...\'\n'
                             'Some examples for single filters:\n'
                             '  \'["gaussian", {"sigma": 2}]\'\n'
                             '  \'["gaussian_gradient_magnitude", {"sigma": 2}]\'\n'
                             '  \'["clahe", {"tile_grid_in_pixels": true, "tile_grid_size": [63,63]}]\'\n'
                             '  \'["vsnr", {"maxit": 20, "filters": {"name": "Gabor", "sigma": [2, 35], "theta": 0, "noise_level": 0.5}}]\'')
    parser.add_argument('-ff', '--filters_file', type=str, default=None,
                        help='*.json file defining the filters as described in the --filters argument. \n'
                             'Using a parameter file should be preferred especially for filters with complex arguments '
                             '(CLAHE, VSNR) or for long filter pipelines. ')
    parser.add_argument('--save_intermediates', action='store_true',
                        help='Saves intermediate files to a directory alongside the filtered result. Directory name is the basename of the output_filepath.')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    input_filepath = args.input_filepath
    out_filepath = args.out_filepath
    filters = args.filters
    filters_file = args.filters_file
    verbose = args.verbose

    if not (bool(filters is None) ^ bool(filters_file is None)):
        raise(ValueError('Supply either --filters or --filters_file and not both!'))
    if filters_file is not None:
        with open(filters_file, mode='r') as f:
            filters = json.load(f)

    from squirrel.workflows.image import filter_2d_workflow

    filter_2d_workflow(
        input_filepath,
        out_filepath,
        filters=filters,
        verbose=verbose
    )
