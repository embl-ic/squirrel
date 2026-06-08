
def invert_slices():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Invert slices within a tif stack or h5 dataset',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('in_path', type=str,
                        help='Input folder containing a tif stack or filepath of h5 container')
    parser.add_argument('out_path', type=str,
                        help='Output folder for tif stack or filepath for h5 container where the results will '
                             'be written to')
    parser.add_argument('--in_pattern', type=str, default='*.tif',
                        help='File patter to search for within the input folder; default = "*.tif"; used if in_path is '
                             'tif stack')
    parser.add_argument('--in_key', type=str, default='data',
                        help='Internal path of the input dataset; default="data"; used if in_path is h5 filepath')
    parser.add_argument('--out_key', type=str, default='data',
                        help='Internal path of the output dataset; default="data"; used if out_path is h5 filepath')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Will process and write data in batches (more memory efficient); default=None')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='The number of cores to use for processing')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    in_path = args.in_path
    out_path = args.out_path
    in_pattern = args.in_pattern
    in_key = args.in_key
    out_key = args.out_key
    batch_size = args.batch_size
    n_workers = args.n_workers
    verbose = args.verbose

    from squirrel.workflows.volume import invert_slices_workflow

    invert_slices_workflow(
        in_path,
        out_path,
        in_pattern=in_pattern,
        in_key=in_key,
        out_key=out_key,
        batch_size=batch_size,
        n_workers=n_workers,
        verbose=verbose
    )


def normalize_slices():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Normalizes slices within a tif stack or h5 dataset',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('in_path', type=str,
                        help='Input folder containing a tif stack or filepath of h5 container')
    parser.add_argument('out_path', type=str,
                        help='Output folder for tif stack or filepath for h5 container where the results will '
                             'be written to')
    parser.add_argument('--in_pattern', type=str, default='*.tif',
                        help='File patter to search for within the input folder; default = "*.tif"; used if in_path is '
                             'tif stack')
    parser.add_argument('--in_key', type=str, default='data',
                        help='Internal path of the input dataset; default="data"; used if in_path is h5 filepath')
    parser.add_argument('--out_key', type=str, default='data',
                        help='Internal path of the output dataset; default="data"; used if out_path is h5 filepath')
    parser.add_argument('--dilate_background', type=int, default=0,
                        help='Dilate the background before computing data quantiles. default=0 (off)')
    parser.add_argument('--quantiles', type=float, nargs=2, default=(0.1, 0.9),
                        help='Lower and upper quantile of the gray value spectrum that is used to normalize;'
                             'default=(0.1, 0.9)')
    parser.add_argument('--anchors', type=float, nargs=2, default=(0.2, 0.8),
                        help='The lower and upper quantiles of the input gray value spectrum are transferred to these '
                             'relative values; default=(0.2, 0.8)')
    parser.add_argument('--keep_zeros', action='store_true',
                        help='All zero-value pixels are kept at zero')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='The number of cores to use for processing')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    in_path = args.in_path
    out_path = args.out_path
    in_pattern = args.in_pattern
    in_key = args.in_key
    out_key = args.out_key
    dilate_background = args.dilate_background
    quantiles = args.quantiles
    anchors = args.anchors
    keep_zeros = args.keep_zeros
    n_workers = args.n_workers
    verbose = args.verbose

    from squirrel.workflows.normalization import normalize_slices_workflow

    normalize_slices_workflow(
        in_path,
        out_path,
        in_pattern=in_pattern,
        in_key=in_key,
        out_key=out_key,
        dilate_background=dilate_background,
        quantiles=quantiles,
        anchors=anchors,
        keep_zeros=keep_zeros,
        n_workers=n_workers,
        verbose=verbose
    )


def adjust_greyscale():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Normalizes slices within a tif stack or h5 dataset',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('in_path', type=str,
                        help='Input folder containing a tif stack or filepath of h5 container')
    parser.add_argument('out_path', type=str,
                        help='Output folder for tif stack or filepath for h5 container where the results will '
                             'be written to')
    parser.add_argument('--in_pattern', type=str, default='*.tif',
                        help='File patter to search for within the input folder; default = "*.tif"; used if in_path is '
                             'tif stack')
    parser.add_argument('--in_key', type=str, default=None,
                        help='Internal path of the input dataset; default="data" in_path is h5 filepath')
    parser.add_argument('--out_key', type=str, default='data',
                        help='Internal path of the output dataset; default="data"; used if out_path is h5 filepath')
    parser.add_argument('--greys_in', nargs=2, type=int, default=None,
                        help='Two grey levels of the input which will be mapped to greys_out; '
                             'default=None, minimum and maximum value within each slices will be used')
    parser.add_argument('--greys_out', nargs=2, type=int, default=None,
                        help='Two grey levels of the output; '
                             'default=None, 0 and maximum of the target data type will be used')
    parser.add_argument('--dtype_out', type=str, default='uint8',
                        help='Data type of the output. ')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='The number of cores to use for processing')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    in_path = args.in_path
    out_path = args.out_path
    in_pattern = args.in_pattern
    in_key = args.in_key
    out_key = args.out_key
    greys_in = args.greys_in
    greys_out = args.greys_out
    dtype_out = args.dtype_out
    n_workers = args.n_workers
    verbose = args.verbose

    from squirrel.workflows.normalization import adjust_greyscale_workflow

    adjust_greyscale_workflow(
        in_path,
        out_path,
        in_pattern=in_pattern,
        in_key=in_key,
        out_key=out_key,
        greys_in=greys_in,
        greys_out=greys_out,
        dtype_out=dtype_out,
        n_workers=n_workers,
        verbose=verbose
    )


def merge_tif_stacks():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Merge two tif stacks',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('stack_folders', nargs='+', type=str,
                        help='List of input folders')
    parser.add_argument('out_folder', type=str,
                        help='Output folder where the results will be written to')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='File pattern for globbing the input stack; default="*.tif"')
    parser.add_argument('--out_pattern', type=str, default='slice_{:05d}.tif',
                        help='File format string defining the output filenames; default="slice_{:05d}.tif')
    parser.add_argument('--pad_canvas', action='store_true',
                        help='For non-equal canvas sizes this can be used to generate a common canvas size for the '
                             'entire result stack')
    parser.add_argument('--inconsistent_shapes', action='store_true',
                        help='Enable this if tif slices within one stack may have different shapes;'
                             'This setting massively increases computation time!')
    parser.add_argument('--add_shifts', type=str, nargs='+', default=None,
                        help='Add a shift to each stack, supply two values for each stack separated by comma; '
                             'default=None\n'
                             'Example: --add_shifts "0,0" "10,20" "(-20,30)"\n'
                             'Note: for negative values you must use parenthesis')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    stack_folders = args.stack_folders
    out_folder = args.out_folder
    pattern = args.pattern
    out_pattern = args.out_pattern
    pad_canvas = args.pad_canvas
    inconsistent_shapes = args.inconsistent_shapes
    add_shifts = args.add_shifts
    verbose = args.verbose

    add_shifts = [[int(n) for n in s.strip("()").split(",")] for s in add_shifts]

    from squirrel.workflows.convert import merge_tif_stacks_workflow

    merge_tif_stacks_workflow(
        stack_folders,
        out_folder,
        pattern=pattern,
        out_pattern=out_pattern,
        pad_canvas=pad_canvas,
        inconsistent_shapes=inconsistent_shapes,
        add_shifts=add_shifts,
        verbose=verbose
    )


def compress_tif_stack():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Create a new compressed copy of a tif stack',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('in_folder', type=str,
                        help='Input folder containing a tif stack')
    parser.add_argument('out_folder', type=str,
                        help='Output folder where the results will be written to')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='File pattern to search for within the input folder; default = "*.tif"')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    in_folder = args.in_folder
    out_folder = args.out_folder
    pattern = args.pattern
    verbose = args.verbose

    from squirrel.workflows.convert import compress_tif_stack_workflow

    compress_tif_stack_workflow(
        in_folder,
        out_folder,
        pattern=pattern,
        verbose=verbose
    )


def crop_from_stack():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Crops a region of interest from a stack and saves it as tif or h5',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('stack_path', type=str,
                        help='Path of the input stack')
    parser.add_argument('out_path', type=str,
                        help='Output location. Must be either a directory or an h5 file name. '
                             'Will be created if not existing')
    parser.add_argument('--roi', type=int, nargs=6, default=None,
                        metavar=('Z', 'Y', 'X', 'D', 'H', 'W'),
                        help='Region of interest to crop, given in voxels\n'
                             'If "D", "H", or "W" is set to 0, the full extent along the respective axis will be used\n'
                             'default=None which will be interpreted as using the full stack')
    parser.add_argument('--key', type=str, default='data',
                        help='For h5 or ome.zarr input stacks this key is used to locate the dataset inside the stack '
                             'location')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='File pattern to search for within the input folder; default = "*.tif"')
    parser.add_argument('--reslice_sample', action='store_true',
                        help='Crops yz slices along the x-axis; --roi is ignored')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    stack_path = args.stack_path
    out_path = args.out_path
    roi = args.roi
    key = args.key
    pattern = args.pattern
    reslice_sample = args.reslice_sample
    verbose = args.verbose

    from squirrel.workflows.volume import crop_from_stack_workflow

    crop_from_stack_workflow(
        stack_path,
        out_path,
        roi=roi,
        key=key,
        pattern=pattern,
        reslice_sample=reslice_sample,
        verbose=verbose
    )


def stack_calculator():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Performs mathematical operations on two image stacks. E.g. pairwise stack_a + stack_b',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('stack_paths', type=str, nargs=2,
                        # metavar=('A', 'B'),
                        help='Paths of the two input stacks')
    parser.add_argument('out_path', type=str,
                        help='Output location. Must be either a directory or an h5 file name. '
                             'Will be created if not existing')
    parser.add_argument('--keys', type=str, default=('data', 'data'), nargs=2,
                        metavar=('A', 'B'),
                        help='For h5 or ome.zarr input stacks this key is used to locate the dataset inside the stack '
                             'location; default=("data", "data"); '
                             'Input an empty string if not applicable for one of the stacks, '
                             'e.g. `--keys "" "data_b"` or `--keys "data_a" ""`')
    parser.add_argument('--patterns', type=str, default=('*.tif', '*.tif'), nargs=2,
                        metavar=('A', 'B'),
                        help='File pattern to search for within the input folder; default = ("*.tif", "*.tif"); '
                             'Input an empty string if not applicable for one of the stacks (see --keys)')
    parser.add_argument('--operation', type=str, default='add',
                        help='Mathematical operation to perform on image slice pairs; default="add"; '
                             'possible values: ("add", "subtract", "multiply", "divide", "min", "max", "average"')
    parser.add_argument('--target_dtype', type=str, default=None,
                        help='If set, the result will be casted to the respective data type')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of CPUs to use')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    stack_paths = args.stack_paths
    out_path = args.out_path
    keys = args.keys
    patterns = args.patterns
    operation = args.operation
    target_dtype = args.target_dtype
    n_workers = args.n_workers
    verbose = args.verbose

    from squirrel.workflows.volume import stack_calculator_workflow

    stack_calculator_workflow(
        stack_paths,
        out_path,
        keys=keys,
        patterns=patterns,
        operation=operation,
        target_dtype=target_dtype,
        n_workers=n_workers,
        verbose=verbose
    )


def axis_median_filter():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Performs a median filtering on an image stack along the given axis',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('stack', type=str,
                        help='Path of the input stack')
    parser.add_argument('out_path', type=str,
                        help='Output location. Must be either a directory or an h5 file name. '
                             'Will be created if not existing')
    parser.add_argument('--key', type=str, default='data',
                        help='For h5 or ome.zarr input stacks this key is used to locate the dataset inside the stack '
                             'location; default="data"')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='File pattern to search for within the input folder; default = "*.tif"')
    parser.add_argument('--median_radius', type=int, default=2,
                        help='Radius of the median filter. The window for the filter will be 2*radius+1; default=2')
    parser.add_argument('--axis', type=int, default=0,
                        help='Axis along which the filter will be applied; default=0 (z-axis)')
    parser.add_argument('--operation', type=str, default=None,
                        help='Operation performed with input and result; default=None \n'
                             ' - None: output = result; dtype = float \n'
                             ' - "difference": output = input - result; dtype: float \n'
                             ' - "difference-clip": output = clip(input - result); dtype: dtype(input)')
    parser.add_argument('--z_range', type=int, nargs=2, default=None,
                        help='Use certain slices of the stack only; Defaults to the entire stack')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Defines the number of slices per batch (decreases memory requirement); default=None')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of CPUs to use; Parallelization is implemented over the slices within one batch;'
                             'Hence, n_workers > batch_size does not decrease run-time')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    stack = args.stack
    out_path = args.out_path
    key = args.key
    pattern = args.pattern
    median_radius = args.median_radius
    axis = args.axis
    operation = args.operation
    z_range = args.z_range
    batch_size = args.batch_size
    n_workers = args.n_workers
    verbose = args.verbose

    from squirrel.workflows.volume import axis_median_filter_workflow

    axis_median_filter_workflow(
        stack,
        out_path,
        key=key,
        pattern=pattern,
        median_radius=median_radius,
        axis=axis,
        operation=operation,
        z_range=z_range,
        batch_size=batch_size,
        n_workers=n_workers,
        verbose=verbose,
    )


def get_label_list():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs numpy unique on a large volume dataset',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_path', type=str,
                        help='Path of the input stack')
    parser.add_argument('--key', type=str, default=None,
                        help='For h5 or ome.zarr input stacks this key is used to locate the dataset inside the stack '
                             'location; default=None which will be interpreted as "s0" for ome.zarr, "data" for h5 and "setup0/timepoint0/s0" for n5')
    parser.add_argument('--pattern', type=str, default=None,
                        help='File pattern to search for within the input folder; default=None (which is interpreted as "*.tif")')
    parser.add_argument('--out_json', type=str, default=None,
                        help='This will trigger writing an output file in json format containing the unique labels; '
                             'default=None will only write it to console')
    parser.add_argument('--z_batch_size', type=int, default=1,
                        help='Defines the number of slices per batch (decreases memory requirement); default=1')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of CPUs to use; Parallelization is implemented over the batches')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    input_path = args.input_path
    key = args.key
    pattern = args.pattern
    out_json = args.out_json
    z_batch_size = args.z_batch_size
    n_workers = args.n_workers
    verbose = args.verbose

    from squirrel.workflows.volume import get_label_list_workflow

    get_label_list_workflow(
        input_path,
        key=key,
        pattern=pattern,
        out_json=out_json,
        z_batch_size=z_batch_size,
        n_workers=n_workers,
        verbose=verbose,
    )


def tif_nearest_scaling():
    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Scales a tif stack using nearest interpolation',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_dirpath', type=str,
                        help='Path of the input tif stack')
    parser.add_argument('output_dirpath', type=str,
                        help='Path of the output tif stack')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='File pattern to search for within the input folder; default="*.tif"')
    parser.add_argument('--scale_factors', type=float, nargs=3, default=[1., 1., 1.],
                        help='Scale factors in z, y, and x; default=[1., 1., 1.], i.e. no scaling')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of CPUs to use; Parallelization is implemented over the batches')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    input_dirpath = args.input_dirpath
    output_dirpath = args.output_dirpath
    pattern = args.pattern
    scale_factors = args.scale_factors
    n_workers = args.n_workers
    verbose = args.verbose

    from squirrel.workflows.volume import tif_nearest_scaling_workflow

    tif_nearest_scaling_workflow(
        input_dirpath,
        output_dirpath,
        pattern=pattern,
        scale_factors=scale_factors,
        n_workers=n_workers,
        verbose=verbose,
    )


def estimate_crop_xy():
    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Estimates a the minimum xy-bounds that contains the full stack',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_path', type=str,
                        help='Path of the input stack')
    parser.add_argument('--key', type=str, default=None,
                        help='For h5 or ome.zarr input stacks this key is used to locate the dataset inside the stack '
                             'location; default=None which will be interpreted as "s0" for ome.zarr, "data" for h5 and "setup0/timepoint0/s0" for n5')
    parser.add_argument('--pattern', type=str, default=None,
                        help='File pattern to search for within the input folder; default=None (which is interpreted as "*.tif")')
    parser.add_argument('-pad', '--padding', type=int, default=64,
                        help='Adds a certain amount of buffer to the output bounds')
    parser.add_argument('-out', '--out_image', type=str, default='max_projection.png',
                        help='Saves the maximum projection as a tif image; defaults to "max_projection.tif"')
    parser.add_argument('-samples', '--number_of_samples', type=int, default=16,
                        help='The number of slices used to generate the maximum projection; default=16')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    input_path = args.input_path
    key = args.key
    pattern = args.pattern
    padding = args.padding
    out_image = args.out_image
    number_of_samples = args.number_of_samples
    verbose = args.verbose

    from squirrel.workflows.volume import estimate_crop_xy_workflow

    estimate_crop_xy_workflow(
        input_path,
        key=key,
        pattern=pattern,
        padding=padding,
        out_image=out_image,
        number_of_samples=number_of_samples,
        verbose=verbose
    )


def filter_2d():
    # ----------------------------------------------------
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description='Performs a 2D filter on each slice of an image stack',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_path', type=str,
                        help='Path of the input stack')
    parser.add_argument('out_path', type=str,
                        help='Output folder for tif stack or filepath for h5 container where the results will be '
                             'written to')
    parser.add_argument('--key', type=str, default=None,
                        help='For h5 or ome.zarr input stacks this key is used to locate the dataset inside the stack '
                             'location; default=None which will be interpreted as "s0" for ome.zarr, '
                             '"data" for h5 and "setup0/timepoint0/s0" for n5')
    parser.add_argument('--pattern', type=str, default=None,
                        help='File pattern to search for within the input folder; '
                             'default=None (which is interpreted as "*.tif")')
    parser.add_argument('--out_key', type=str, default=None,
                        help='For h5 and ome.zarr output paths')
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
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Will process and write data in batches (more memory efficient); default=None')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='The number of cores to use for processing')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    input_path = args.input_path
    out_path = args.out_path
    key = args.key
    pattern = args.pattern
    out_key = args.out_key
    filters = args.filters
    filters_file = args.filters_file
    batch_size = args.batch_size
    n_workers = args.n_workers
    verbose = args.verbose

    if not (bool(filters is None) ^ bool(filters_file is None)):
        raise(ValueError('Supply either --filters or --filters_file and not both!'))
    if filters_file is not None:
        with open(filters_file, mode='r') as f:
            filters = json.load(f)

    from squirrel.workflows.volume import filter_2d_workflow

    filter_2d_workflow(
        input_path,
        out_path,
        key=key,
        pattern=pattern,
        out_key=out_key,
        filters=filters,
        batch_size=batch_size,
        n_workers=n_workers,
        verbose=verbose
    )


def stack_to_consistent_shapes():
    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Takes an image stack (tif or png) with inconsistent slice shapes and writes a stack with '
                    'consistent slice shapes',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('stack_dirpath', type=str,
                        help='Path of the input stack')
    parser.add_argument('output_path', type=str,
                        help='Output folder for tif stack where the results will be written to')
    parser.add_argument('--pattern', type=str, default=None,
                        help='File pattern to search for within the input folder; '
                             'default="*.tif"'
                             'Is also used to determine filetype: "*.png" or "*.PNG" can be used to read png stack')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    stack_dirpath = args.stack_dirpath
    output_path = args.output_path
    pattern = args.pattern
    verbose = args.verbose

    from squirrel.workflows.convert import stack_to_consistent_shapes_workflow
    stack_to_consistent_shapes_workflow(
        stack_dirpath,
        output_path,
        pattern,
        verbose=verbose
    )
