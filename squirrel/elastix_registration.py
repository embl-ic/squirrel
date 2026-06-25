import os


def register_z_chunks():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs Simple Elastix on chunks along the z-axis of a dataset individually.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('moving_filepath', type=str,
                        help='Input filepath for the moving volume (nii or h5)')
    parser.add_argument('fixed_filepath', type=str,
                        help='Input filepath for the fixed volume (nii or h5)')
    parser.add_argument('out_path', type=str,
                        help='Output folder for all results files (must be empty or not existing)')
    parser.add_argument('--moving_key', type=str, default='data',
                        help='Internal path of the moving input; default="data"; used if moving_filepath is h5 file')
    parser.add_argument('--fixed_key', type=str, default='data',
                        help='Internal path of the fixed input; default="data"; used if fixed_filepath is h5 file')
    parser.add_argument('--transform', type=str, default='affine',
                        help='The transformation. Can be "affine" (default) or "rigid" ')
    parser.add_argument('--z_chunk_size', type=int, default=16,
                        help='Size of the z-chunks')
    parser.add_argument('-auto_init', '--automatic_transform_initialization', action='store_true',
                        help='Triggers the automatic transform initialization from Elastix; Not needed if both images'
                             'are already close')
    parser.add_argument('-napari', '--view_results_in_napari', action='store_true',
                        help='Show the results in the napari viewer')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    moving_filepath = args.moving_filepath
    fixed_filepath = args.fixed_filepath
    out_path = args.out_path
    moving_key = args.moving_key
    fixed_key = args.fixed_key
    z_chunk_size = args.z_chunk_size
    transform = args.transform
    automatic_transform_initialization = args.automatic_transform_initialization
    view_results_in_napari = args.view_results_in_napari
    verbose = args.verbose

    from squirrel.workflows.elastix import register_z_chunks

    register_z_chunks(
        moving_filepath,
        fixed_filepath,
        out_path,
        moving_key=moving_key,
        fixed_key=fixed_key,
        z_chunk_size=z_chunk_size,
        transform=transform,
        automatic_transform_initialization=automatic_transform_initialization,
        view_results_in_napari=view_results_in_napari,
        verbose=verbose
    )


def register_with_elastix():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs a registration with Elastix',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('moving_filepath', type=str,
                        help='Input tiff file for the moving image')
    parser.add_argument('fixed_filepath', type=str,
                        help='Input tiff file for the fixed image')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the transformation (*.json)')
    parser.add_argument('--out_img_filepath', type=str, default=None,
                        help='Output filepath for transformed image')
    parser.add_argument('--transform', type=str, default='translation',
                        help='The transformation; default="translation"')
    parser.add_argument('--auto_mask', type=str, default=None,
                        help='Automatically generates a mask for fixed and moving image; '
                             'default=None; ["non-zero", "variance"]')
    parser.add_argument('--number_of_spatial_samples', type=int, default=None,
                        help='Elastix parameter')
    parser.add_argument('--maximum_number_of_iterations', type=int, default=None,
                        help='Elastix parameter')
    parser.add_argument('--number_of_resolutions', type=int, default=None,
                        help='Elastix parameter')
    parser.add_argument('--initialize_offsets_method', type=str, default=None,
                        help='Method to solve big jumps: ["xcorr", "init_elx"]\n'
                             '  "xcorr": Big jumps are detected by the intersection over union of the data in adjacent '
                             'slices. \n'
                             '      Correction is performed by cross-correlation\n'
                             '  "init_elx": Using a highly binned version of the data, initial shifts are performed '
                             '(grid). \n'
                             '      The shifted moving image is used as input for elastix registration and the '
                             'resulting alignment measured by Mutual Information (MI). \n'
                             '      When MI < mi_thresh is reached or all positions of the grid are tested, the best '
                             'offset is used to initialize the alignment')
    parser.add_argument('--initialize_offsets_kwargs', type=str, nargs='+', default=(),
                        help='Arguments for the respective initialization method. Syntax: "key:value"\n'
                             '  defaults for the respective method:\n'
                             '      "xcorr":\n'
                             '          iou_thresh:0.5\n'
                             '      "init_elx:\n'
                             '          binning:32\n'
                             '          spacing:256\n'
                             '          elx_binning:4\n'
                             '          elx_max_iters:32\n'
                             '          mi_thresh:-0.8')
    parser.add_argument('--gaussian_sigma', type=float, default=0.,
                        help='Perform a gaussian filter before registration')
    parser.add_argument('--use_clahe', action='store_true',
                        help='CLAHE filter before registration')
    parser.add_argument('--use_edges', action='store_true',
                        help='Computes edges before registration using a sobel filter')
    parser.add_argument('--parameter_map', type=str, default=None,
                        help='Elastix parameter map file')
    parser.add_argument('--n_workers', type=int, default=os.cpu_count(),
                        help='The number of cores to use for processing')
    parser.add_argument('--debug_dirpath', type=str, default=None,
                        help='Saves intermediate files for debugging')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    moving_filepath = args.moving_filepath
    fixed_filepath = args.fixed_filepath
    out_filepath = args.out_filepath
    out_img_filepath = args.out_img_filepath
    transform = args.transform
    auto_mask = args.auto_mask
    number_of_spatial_samples = args.number_of_spatial_samples
    maximum_number_of_iterations = args.maximum_number_of_iterations
    number_of_resolutions = args.number_of_resolutions
    initialize_offsets_method = args.initialize_offsets_method
    initialize_offsets_kwargs = args.initialize_offsets_kwargs
    gaussian_sigma = args.gaussian_sigma
    use_clahe = args.use_clahe
    use_edges = args.use_edges
    parameter_map = args.parameter_map
    debug_dirpath = args.debug_dirpath
    n_workers = args.n_workers
    verbose = args.verbose

    if verbose:
        print(f'initialize_offsets_kwargs = {initialize_offsets_kwargs}')

    from squirrel.library.string_conversion import parse_kwargs_list
    initialize_offsets_kwargs = parse_kwargs_list(initialize_offsets_kwargs)

    if verbose:
        print(f'initialize_offsets_kwargs = {initialize_offsets_kwargs}')

    from squirrel.workflows.elastix import register_with_elastix_workflow
    register_with_elastix_workflow(
        moving_filepath,
        fixed_filepath,
        out_filepath,
        out_img_filepath=out_img_filepath,
        transform=transform,
        auto_mask=auto_mask,
        number_of_spatial_samples=number_of_spatial_samples,
        maximum_number_of_iterations=maximum_number_of_iterations,
        number_of_resolutions=number_of_resolutions,
        initialize_offsets_method=initialize_offsets_method,
        initialize_offsets_kwargs=initialize_offsets_kwargs,
        gaussian_sigma=gaussian_sigma,
        use_clahe=use_clahe,
        use_edges=use_edges,
        parameter_map=parameter_map,
        debug_dirpath=debug_dirpath,
        n_workers=n_workers,
        verbose=verbose,
    )


def elastix_on_volume3d():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs Simple Elastix with an affine transform',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('moving_filepath', type=str,
                        help='Input filepath for the moving volume (nii or h5)')
    parser.add_argument('fixed_filepath', type=str,
                        help='Input filepath for the fixed volume (nii or h5)')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for all results files (must be empty or not existing)')
    parser.add_argument('--moving_key', type=str, default='data',
                        help='Internal path of the moving input; default="data"; used if moving_filepath is h5 file')
    parser.add_argument('--fixed_key', type=str, default='data',
                        help='Internal path of the fixed input; default="data"; used if fixed_filepath is h5 file')
    parser.add_argument('--transform', type=str, default='affine',
                        help='The transformation; default="affine"')
    parser.add_argument('-auto_init', '--automatic_transform_initialization', action='store_true',
                        help='Triggers the automatic transform initialization from Elastix; Not needed if both images'
                             'are already close')
    parser.add_argument('--pivot', type=float, nargs=3, default=(0., 0., 0.),
                        help='To where the resulting transformation is centered')
    parser.add_argument('-napari', '--view_results_in_napari', action='store_true',
                        help='Show the results in the napari viewer')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    moving_filepath = args.moving_filepath
    fixed_filepath = args.fixed_filepath
    out_filepath = args.out_filepath
    moving_key = args.moving_key
    fixed_key = args.fixed_key
    transform = args.transform
    automatic_transform_initialization = args.automatic_transform_initialization
    pivot = args.pivot
    view_results_in_napari = args.view_results_in_napari
    verbose = args.verbose

    from squirrel.workflows.elastix import elastix3d

    elastix3d(
        moving_filepath,
        fixed_filepath,
        out_filepath,
        moving_key=moving_key,
        fixed_key=fixed_key,
        transform=transform,
        automatic_transform_initialization=automatic_transform_initialization,
        pivot=pivot,
        view_results_in_napari=view_results_in_napari,
        verbose=verbose
    )


def elastix_slices_to_volume():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Register 2D slices of an image stack to a corresponding volume',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('moving_filepath', type=str,
                        help='Input filepath for the moving volume (nii or h5)')
    parser.add_argument('fixed_filepath', type=str,
                        help='Input filepath for the fixed volume (nii or h5)')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the result volume.')
    parser.add_argument('--moving_key', type=str, default='data',
                        help='Internal path of the moving input; default="data"; used if moving_filepath is h5 file')
    parser.add_argument('--fixed_key', type=str, default='data',
                        help='Internal path of the fixed input; default="data"; used if fixed_filepath is h5 file')
    parser.add_argument('--transform', type=str, default='affine',
                        help='The transformation; default="affine"')
    parser.add_argument('-auto_init', '--automatic_transform_initialization', action='store_true',
                        help='Triggers the automatic transform initialization from Elastix; Not needed if both images'
                             'are already close')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    moving_filepath = args.moving_filepath
    fixed_filepath = args.fixed_filepath
    out_filepath = args.out_filepath
    moving_key = args.moving_key
    fixed_key = args.fixed_key
    transform = args.transform
    automatic_transform_initialization = args.automatic_transform_initialization
    verbose = args.verbose

    from squirrel.workflows.elastix import slices_to_volume

    slices_to_volume(
        moving_filepath,
        fixed_filepath,
        out_filepath,
        moving_key=moving_key,
        fixed_key=fixed_key,
        transform=transform,
        automatic_transform_initialization=automatic_transform_initialization,
        verbose=verbose
    )


def amst():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs Alignment to Median Smoothed Template (AMST)',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('pre_aligned_stack', type=str,
                        help='Input filepath for an aligned image stack (h5, ome.zarr or tif stack)')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the transformations (*.json)')
    parser.add_argument('--raw_stack', type=str, default=None,
                        help='Input of the raw stack from which the pre-alignment was derived '
                             '(Note: Not yet implemented)')
    parser.add_argument('--pre_align_key', type=str, default='data',
                        help='Internal path of the input; default="data"; used if stack is h5 or ome.zarr')
    parser.add_argument('--pre_align_pattern', type=str, default='*.tif',
                        help='Used to glob tif files from a tif stack folder; default="*.tif"')
    parser.add_argument('--transform', type=str, default='affine',
                        help='The transformation to use for alignment to the template; default="affine"')
    parser.add_argument('--auto_mask_off', action='store_true',
                        help='Turn off automatic generation of a mask for fixed and moving image')
    # parser.add_argument('--number_of_spatial_samples', type=int, default=None,
    #                     help='Elastix parameter')
    # parser.add_argument('--maximum_number_of_iterations', type=int, default=None,
    #                     help='Elastix parameter')
    # parser.add_argument('--number_of_resolutions', type=int, default=None,
    #                     help='Elastix parameter')
    # parser.add_argument('--pre_fix_big_jumps', action='store_true',
    #                     help='Determines big jumps and fixes them using cross-correlation')
    parser.add_argument('--median_radius', type=int, default=7,
                        help='Radius of the z-median-smoothing used to create the template stack; default=7')
    parser.add_argument('--gaussian_sigma', type=float, default=0.,
                        help='Pre-smooth the images using a gaussian filter; default=0. (no smoothing)')
    parser.add_argument('--z_range', type=int, nargs=2, default=None,
                        help='Use certain slices of the stack only; Defaults to the entire stack')
    parser.add_argument('--elastix_parameter_filepath', type=str, default=None,
                        help='Filepath of an elastix parameter file; default=None')
    parser.add_argument('--crop_to_bounds_off', action='store_true',
                        help='Switches off automated cropping to image bounds')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='The number of cores to use for processing')
    # parser.add_argument('--debug', action='store_true',
    #                     help='Writes elastix registration input images to disk')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    pre_aligned_stack = args.pre_aligned_stack
    out_filepath = args.out_filepath
    raw_stack = args.raw_stack
    pre_align_key = args.pre_align_key
    pre_align_pattern = args.pre_align_pattern
    transform = args.transform
    auto_mask_off = args.auto_mask_off
    median_radius = args.median_radius
    gaussian_sigma = args.gaussian_sigma
    z_range = args.z_range
    elastix_parameter_filepath = args.elastix_parameter_filepath
    crop_to_bounds_off = args.crop_to_bounds_off
    n_workers = args.n_workers
    # debug = args.debug
    verbose = args.verbose

    from squirrel.workflows.amst import amst_workflow

    amst_workflow(
        pre_aligned_stack,
        out_filepath,
        raw_stack=raw_stack,
        pre_align_key=pre_align_key,
        pre_align_pattern=pre_align_pattern,
        transform=transform,
        auto_mask_off=auto_mask_off,
        median_radius=median_radius,
        gaussian_sigma=gaussian_sigma,
        z_range=z_range,
        elastix_parameters=elastix_parameter_filepath,
        crop_to_bounds_off=crop_to_bounds_off,
        n_workers=n_workers,
        # debug=debug,
        verbose=verbose
    )


def elastix_stack_alignment():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs a stack alignment with Elastix',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('stack', type=str,
                        help='Input filepath for the image stack (h5 or tif stack)')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the transformations (*.json)')
    parser.add_argument('--key', type=str, default='data',
                        help='Internal path of the input; default="data"; used if stack is h5 file')
    parser.add_argument('--transform', type=str, default='translation',
                        help='The transformation; default="translation"')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='Used to glob tif files from a tif stack folder; default="*.tif"')
    parser.add_argument('--auto_mask', type=str, default='None',
                        help='Automatically generates a mask for fixed and moving image; '
                             'default=None; ["non-zero", "variance"]')
    parser.add_argument('--number_of_spatial_samples', type=int, default=None,
                        help='Elastix parameter')
    parser.add_argument('--maximum_number_of_iterations', type=int, default=None,
                        help='Elastix parameter')
    parser.add_argument('--number_of_resolutions', type=int, default=None,
                        help='Elastix parameter')
    # parser.add_argument('--extended_output', action='store_true',
    #                     help='Increase information content of the output')
    parser.add_argument('--initialize_offsets_method', type=str, default=None,
                        help='Method to solve big jumps: ["xcorr", "init_elx"]\n'
                             '  "xcorr": Big jumps are detected by the intersection over union of the data in adjacent '
                             'slices. \n'
                             '      Correction is performed by cross-correlation\n'
                             '  "init_elx": Using a highly binned version of the data, initial shifts are performed '
                             '(grid). \n'
                             '      The shifted moving image is used as input for elastix registration and the '
                             'resulting alignment measured by Mutual Information (MI). \n'
                             '      When MI < mi_thresh is reached or all positions of the grid are tested, the best '
                             'offset is used to initialize the alignment')
    parser.add_argument('--initialize_offsets_kwargs', type=str, nargs='+', default=(),
                        help='Arguments for the respective initialization method. Syntax: "key:value"\n'
                             '  defaults for the respective method:\n'
                             '      "xcorr":\n'
                             '          iou_thresh:0.5\n'
                             '      "init_elx:\n'
                             '          binning:32\n'
                             '          spacing:256\n'
                             '          elx_binning:4\n'
                             '          elx_max_iters:32\n'
                             '          mi_thresh:-0.8')
    parser.add_argument('--gaussian_sigma', type=float, default=0.,
                        help='Perform a gaussian filter before registration')
    parser.add_argument('--use_clahe', action='store_true',
                        help='CLAHE filter before registration')
    parser.add_argument('--use_edges', action='store_true',
                        help='Computes edges before registration using a sobel filter')
    parser.add_argument('--parameter_map', type=str, default=None,
                        help='Elastix parameter map file')
    parser.add_argument('--z_range', type=int, nargs=2, default=None,
                        help='Use certain slices of the stack only; Defaults to the entire stack')
    parser.add_argument('--z_step', type=int, default=1,
                        help='Performs an alignment with every n-th slice only.')
    parser.add_argument('--determine_bounds', action='store_true',
                        help='Appends the bounding box of data within each slice to the results metadata. '
                             'Useful for auto-padding later on')
    parser.add_argument('--n_workers', type=int, default=os.cpu_count(),
                        help='The number of cores to use for processing')
    parser.add_argument('--debug', action='store_true',
                        help='Saves intermediate files for debugging')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    stack = args.stack
    out_filepath = args.out_filepath
    key = args.key
    transform = args.transform
    pattern = args.pattern
    auto_mask = args.auto_mask
    number_of_spatial_samples = args.number_of_spatial_samples
    maximum_number_of_iterations = args.maximum_number_of_iterations
    number_of_resolutions = args.number_of_resolutions
    initialize_offsets_method = args.initialize_offsets_method
    initialize_offsets_kwargs = args.initialize_offsets_kwargs
    gaussian_sigma = args.gaussian_sigma
    use_clahe = args.use_clahe
    use_edges = args.use_edges
    parameter_map = args.parameter_map
    z_range = args.z_range
    z_step = args.z_step
    determine_bounds = args.determine_bounds
    n_workers = args.n_workers
    debug = args.debug
    verbose = args.verbose

    def convert_value(v):
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v
    initialize_offsets_kwargs = {
        k: convert_value(v) for k, v in (item.split(':', 1) for item in initialize_offsets_kwargs)
    }

    from squirrel.workflows.elastix import elastix_stack_alignment_workflow

    elastix_stack_alignment_workflow(
        stack,
        out_filepath,
        transform=transform,
        key=key,
        pattern=pattern,
        auto_mask=auto_mask,
        number_of_spatial_samples=number_of_spatial_samples,
        maximum_number_of_iterations=maximum_number_of_iterations,
        number_of_resolutions=number_of_resolutions,
        initialize_offsets_method=initialize_offsets_method,
        initialize_offsets_kwargs=initialize_offsets_kwargs,
        gaussian_sigma=gaussian_sigma,
        use_clahe=use_clahe,
        use_edges=use_edges,
        parameter_map=parameter_map,
        z_range=z_range,
        z_step=z_step,
        determine_bounds=determine_bounds,
        n_workers=n_workers,
        debug=debug,
        verbose=verbose
    )


def stack_alignment_validation():
    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Validation of a stack alignment',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('stack', type=str,
                        help='Input filepath for the image stack (h5 or tif stack)')
    parser.add_argument('out_dirpath', type=str,
                        help='Output directory path for the result plots')
    parser.add_argument('rois', nargs='+', type=str,
                        help='Define one or multiple regions for validation in pixels\n'
                             'Format: "z,y,x,d,h,w"\n'
                             'Use "0" to denote an entire axis, e.g.: "0,100,200,0,64,64" will grab a 64^2 region '
                             '  along the entire z-axis')
    parser.add_argument('--key', type=str, default='data',
                        help='Internal path of the input; default="data"; used if stack is h5 file')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='Used to glob tif files from a tif stack folder; default="*.tif"')
    parser.add_argument('--resolution_yx', type=float, nargs=2, default=(1.0, 1.0),
                        help='xy-resolution used to properly plot the results. Default=(1.0, 1.0)')
    parser.add_argument('--out_name', type=str, default=None,
                        help='An additional name for the result files which is added to the filenames')
    parser.add_argument('--y_max', type=int, default=None,
                        help='Maximum of the y-axis')
    parser.add_argument('--method', type=str, default='elastix',
                        help='The registration method; either "elastix" or "xcorr"; default="elastix"')
    parser.add_argument('--method_kwargs', nargs='+', type=str, default=None,
                        help='Keyword argument of the registration method. \n'
                             '  Format example (for method="elastix"): \n'
                             '    --method_kwargs parameter_map:/path/to/file.txt')
    parser.add_argument('--gaussian_sigma', type=float, default=0.,
                        help='Perform a gaussian filter before registration')
    parser.add_argument('--use_clahe', action='store_true',
                        help='CLAHE filter before registration')
    parser.add_argument('--subtract_average', action='store_true',
                        help='Subtract the local average of the alignment signal to avoid drift effect to influece the '
                             'error estimation')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    stack = args.stack
    out_dirpath = args.out_dirpath
    rois = args.rois
    key = args.key
    pattern = args.pattern
    resolution_yx = args.resolution_yx
    out_name = args.out_name
    y_max = args.y_max
    method = args.method
    method_kwargs = args.method_kwargs
    gaussian_sigma = args.gaussian_sigma
    use_clahe = args.use_clahe
    subtract_average = args.subtract_average
    verbose = args.verbose

    from squirrel.library.argument_parsing import column_separated_inputs_to_dict
    method_kwargs = column_separated_inputs_to_dict(method_kwargs)

    from squirrel.workflows.elastix import stack_alignment_validation_workflow

    from squirrel.library.string_conversion import str2values
    from squirrel.library.rois import list2roi
    rois = [list2roi(str2values(roi, dtype='int')) for roi in rois]

    stack_alignment_validation_workflow(
        stack,
        out_dirpath,
        rois,
        key=key,
        pattern=pattern,
        resolution_yx=resolution_yx,
        out_name=out_name,
        y_max=y_max,
        method=method,
        method_kwargs=method_kwargs,
        gaussian_sigma=gaussian_sigma,
        use_clahe=use_clahe,
        subtract_average=subtract_average,
        verbose=verbose,
    )


def make_elastix_default_parameter_file():
    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Generates a default parameter file for Elastix registration',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('out_filepath', type=str,
                        help='Output directory path for the result plots')
    parser.add_argument('--transform', type=str, default='translation',
                        help='One of the available transforms: ["translation", "affine", "bspline", "amst-bspline"]; '
                             'default="translation"')
    parser.add_argument('-elx', '--elastix_parameters', type=str, nargs='+', default=None,
                        help='Specify any elastix parameter that needs to deviate from the default; default=None\n'
                             'Format example: \n'
                             '    -elx FinalGridSpacingInPhysicalUnits:128 GridSpacingSchedule:2.0,1.4,1.2,1.0')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    out_filepath = args.out_filepath
    transform = args.transform
    elastix_parameters = args.elastix_parameters
    verbose = args.verbose

    if verbose:
        print(f'out_filepath = {out_filepath}')
        print(f'transform = {transform}')
        print(f'elastix_parameters = {elastix_parameters}')

    from squirrel.workflows.elastix import make_elastix_default_parameter_file_workflow
    make_elastix_default_parameter_file_workflow(
        out_filepath,
        transform=transform,
        elastix_parameters=elastix_parameters,
        verbose=verbose
    )


def apply_multi_step_stack_alignment():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Applies an affine transformation on a volume',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('image_stack', type=str,
                        help='Input filepath for the image stack (h5 or tif stack)')
    parser.add_argument('transform_paths', type=str, nargs='+',
                        help='Json file or folders containing the transformations for each slice')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the result file (only h5 for now)')
    parser.add_argument('--key', type=str, default='data',
                        help='Internal path of the input; default="data"; used if stack is h5 file')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='Used to glob tif files from a tif stack folder; default="*.tif"')
    parser.add_argument('--auto_pad', action='store_true',
                        help='Automatically adjust the canvas size of the output stack to best fit the data')
    parser.add_argument('--target_image_shape', type=int, nargs=3, default=None,
                        help='Pre-define a stack shape for the output stack; default=None')
    parser.add_argument('--z_range', type=int, nargs=2, default=None,
                        help='Use certain slices of the stack only; Defaults to the entire stack')
    parser.add_argument('--resample_interpolator', type=str, default=None,
                        help='Select the interpolator; Default = None; Examples:\n'
                             '  "none": evaluates to None -> Default\n'
                             '  "nearest" / "FinalNearestNeighborInterpolator": nearest interpolation')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='The number of cores to use for processing')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    image_stack = args.image_stack
    transform_paths = args.transform_paths
    out_filepath = args.out_filepath
    key = args.key
    pattern = args.pattern
    auto_pad = args.auto_pad
    target_image_shape = args.target_image_shape
    z_range = args.z_range
    resample_interpolator = args.resample_interpolator
    n_workers = args.n_workers
    verbose = args.verbose

    from squirrel.workflows.elastix import apply_multi_step_stack_alignment_workflow

    apply_multi_step_stack_alignment_workflow(
        image_stack,
        transform_paths,
        out_filepath,
        key=key,
        pattern=pattern,
        auto_pad=auto_pad,
        target_image_shape=target_image_shape,
        z_range=z_range,
        write_result=True,
        resample_interpolator=resample_interpolator,
        n_workers=n_workers,
        verbose=verbose
    )


def get_elastix_transforms_substack():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Extracts a substack of the elastix transforms for a stack alignment',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('transform_stack', type=str,
                        help='Input filepath for the elastix transform stack (h5 file)')
    parser.add_argument('out_dirpath', type=str,
                        help='Output directory path for the result files')
    parser.add_argument('--z_range', type=int, nargs=2, default=None,
                        help='Use certain slices of the stack only; Defaults to the entire stack')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    transform_stack = args.transform_stack
    out_dirpath = args.out_dirpath
    z_range = args.z_range
    verbose = args.verbose

    from squirrel.workflows.elastix import get_elastix_transforms_substack_workflow

    get_elastix_transforms_substack_workflow(
        transform_stack,
        out_dirpath,
        z_range=z_range,
        verbose=verbose
    )


# def stitch_parts():
#
#     # ----------------------------------------------------
#     import argparse
#
#     parser = argparse.ArgumentParser(
#         description='Computes the required transform to stitch two parts of a stack',
#         formatter_class=argparse.RawTextHelpFormatter
#     )
#     parser.add_argument('image_stacks', nargs=2, type=str,
#                         help='Input filepath for the image stacks (h5, n5, ome.zarr or tif stack)')
#     parser.add_argument('out_filepath', type=str,
#                         help='Output filepath for the result file (only h5 for now)')
#     parser.add_argument('--keys', nargs=2, type=str, default=(None, None),
#                         help='Internal path of the input; default="data"; used if stack is h5 file')
#     parser.add_argument('--patterns', nargs=2, type=str, default=('*.tif', '*.tif'),
#                         help='Used to glob tif files from a tif stack folder; default="*.tif"')
#     parser.add_argument('--n_workers', type=int, default=1,
#                         help='The number of cores to use for processing')
#     parser.add_argument('-v', '--verbose', action='store_true')
#
#     args = parser.parse_args()
#     image_stacks = args.image_stacks
#     out_filepath = args.out_filepath
#     keys = args.keys
#     patterns = args.patterns
#     n_workers = args.n_workers
#     verbose = args.verbose
#
#     from squirrel.workflows.elastix import stitch_parts_workflow
#     stitch_parts_workflow(
#         image_stacks,
#         out_filepath,
#         keys=keys,
#         patterns=patterns,
#         n_workers=n_workers,
#         verbose=verbose
#     )


if __name__ == '__main__':

    # elastix_stack_alignment()
    # amst()
    # stack_alignment_validation()
    apply_multi_step_stack_alignment()
