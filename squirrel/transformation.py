
def affine_on_volume():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Applies an affine transformation on a volume',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('image_filepath', type=str,
                        help='Input filepath for the moving volume (nii or h5)')
    parser.add_argument('transform_filepath', type=str,
                        help='Json file containing the transformation')
    parser.add_argument('-o', '--out_filepath', type=str, default=None,
                        help='Output filepath for the result file (only h5 for now)')
    parser.add_argument('--image_key', type=str, default='data',
                        help='Internal path of the moving input; default="data"; used if moving_filepath is h5 file')
    parser.add_argument('--no_offset_to_center', action='store_true',
                        help="If set, the image is rotated around it's origin")
    # parser.add_argument('--pivot', type=float, default=None, nargs=3,
    #                     help='Center point location')
    parser.add_argument('--scale_canvas', action='store_true',
                        help='Scale the image canvas to match the scaling of the data. Beware of shear and rotation!')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    image_filepath = args.image_filepath
    transform_filepath = args.transform_filepath
    out_filepath = args.out_filepath
    image_key = args.image_key
    no_offset_to_center = args.no_offset_to_center
    # pivot = args.pivot
    scale_canvas = args.scale_canvas
    verbose = args.verbose

    from squirrel.workflows.transformation import apply_affine
    apply_affine(
        image_filepath,
        transform_filepath,
        out_filepath=out_filepath,
        image_key=image_key,
        no_offset_to_center=no_offset_to_center,
        # pivot=pivot,
        scale_canvas=scale_canvas,
        verbose=verbose
    )


def apply_stack_alignment():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Applies an affine transformation on a volume',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('stack', type=str,
                        help='Input filepath for the image stack (h5 or tif stack)')
    parser.add_argument('transform_filepath', type=str,
                        help='Json file containing the transformations for each slice')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the result file (only h5 for now)')
    parser.add_argument('--key', type=str, default='data',
                        help='Internal path of the input; default="data"; used if stack is h5 file')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='Used to glob tif files from a tif stack folder; default="*.tif"')
    parser.add_argument('--no_adding_of_transforms', action='store_true',
                        help='By default each transformation is treated as the dot product of the previous.'
                             'If this flag is set, each transform is applied as it is')
    parser.add_argument('--auto_pad', action='store_true',
                        help='Automatically adjust the canvas size of the output stack to best fit the data')
    parser.add_argument('--stack_shape', type=int, nargs=3, default=None,
                        help='Pre-define a stack shape for the output stack; default=None')
    parser.add_argument('--z_range', type=int, nargs=2, default=None,
                        help='Use certain slices of the stack only; Defaults to the entire stack')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='The number of cores to use for processing')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    stack = args.stack
    transform_filepath = args.transform_filepath
    out_filepath = args.out_filepath
    key = args.key
    pattern = args.pattern
    no_adding_of_transforms = args.no_adding_of_transforms
    auto_pad = args.auto_pad
    stack_shape = args.stack_shape
    z_range = args.z_range
    n_workers = args.n_workers
    verbose = args.verbose

    from squirrel.workflows.transformation import apply_stack_alignment_on_volume_workflow
    apply_stack_alignment_on_volume_workflow(
        stack,
        transform_filepath,
        out_filepath,
        key=key,
        pattern=pattern,
        no_adding_of_transforms=no_adding_of_transforms,
        auto_pad=auto_pad,
        stack_shape=stack_shape,
        z_range=z_range,
        n_workers=n_workers,
        verbose=verbose,
    )


def sequential_affine_on_volume():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Applies an affine transformation on a volume',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('image_filepath', type=str,
                        help='Input filepath for the moving volume (nii or h5)')
    parser.add_argument('transform_filepaths', type=str, nargs='+',
                        help='Json or csv file(s) containing the transformations')
    parser.add_argument('-o', '--out_filepath', type=str, default=None,
                        help='Output filepath for the result file (only h5 for now)')
    parser.add_argument('--image_key', type=str, default='data',
                        help='Internal path of the moving input; default="data"; used if moving_filepath is h5 file')
    parser.add_argument('--no_offset_to_center', action='store_true',
                        help="If set, the image is rotated around it's origin")
    parser.add_argument('--pivot', type=float, default=None, nargs=3,
                        help='Center point location')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    image_filepath = args.image_filepath
    transform_filepaths = args.transform_filepaths
    out_filepath = args.out_filepath
    image_key = args.image_key
    no_offset_to_center = args.no_offset_to_center
    pivot = args.pivot
    verbose = args.verbose

    from squirrel.workflows.transformation import apply_sequential_affine
    apply_sequential_affine(
        image_filepath,
        transform_filepaths,
        out_filepath=out_filepath,
        image_key=image_key,
        no_offset_to_center=no_offset_to_center,
        pivot=pivot,
        verbose=verbose
    )


def decompose_affine_matrix():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Applies an affine transformation on a volume',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('transform_filepath', type=str,
                        help='Json or csv file containing the transformation')
    parser.add_argument('-o', '--out_folder', type=str, default=None,
                        help='Output filepath for the result files')
    parser.add_argument('--shear_to_translation_pivot', type=float, nargs=3, default=None,
                        help='Add a translation to the translation matrix '
                             'that compensates for a missing shear component; '
                             'The three pivot coordinates are the reference position to compute the offset'
                             'that the shear would create')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    transform_filepath = args.transform_filepath
    out_folder = args.out_folder
    shear_to_translation_pivot = args.shear_to_translation_pivot
    verbose = args.verbose

    from squirrel.workflows.transformation import decompose_affine
    decompose_affine(
        transform_filepath,
        out_folder=out_folder,
        shear_to_translation_pivot=shear_to_translation_pivot,
        verbose=verbose
    )


def average_affine_on_volume():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Applies an average of multiple affine transforms to a volume.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('image_filepath', type=str,
                        help='Input filepath for the moving volume (nii or h5)')
    parser.add_argument('transform_filepath', type=str,
                        help='Json file containing the transformation')
    parser.add_argument('-o', '--out_filepath', type=str, default=None,
                        help='Output filepath for the result file (only h5 for now)')
    parser.add_argument('--image_key', type=str, default='data',
                        help='Internal path of the moving input; default="data"; used if moving_filepath is h5 file')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    image_filepath = args.image_filepath
    transform_filepath = args.transform_filepath
    out_filepath = args.out_filepath
    image_key = args.image_key
    verbose = args.verbose

    from squirrel.workflows.transformation import apply_average_affine
    apply_average_affine(
        image_filepath,
        transform_filepath,
        out_filepath=out_filepath,
        image_key=image_key,
        verbose=verbose
    )


def apply_z_chunks_to_volume():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Applies an average of multiple affine transforms to a volume.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('image_filepath', type=str,
                        help='Input filepath for the moving volume (nii or h5)')
    parser.add_argument('transforms', type=str,
                        help='Json file containing the transformation')
    parser.add_argument('chunk_distance', type=int,
                        help='The distance in pixels of the sub-stacks')
    parser.add_argument('-o', '--out_filepath', type=str, default=None,
                        help='Output filepath for the result file (only h5 for now)')
    parser.add_argument('--image_key', type=str, default='data',
                        help='Internal path of the moving input; default="data"; used if moving_filepath is h5 file')
    parser.add_argument('--apply', type=str, nargs='+', default=('rotate', 'scale', 'translate'),
                        help='Applies the defined transform components to the volume')
    parser.add_argument('--pivot', type=float, nargs=3, default=(0., 0., 0.),
                        help='Point around which scaling is centered. Caution: This does not work for rotation (yet?)')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    image_filepath = args.image_filepath
    transforms = args.transforms
    chunk_distance = args.chunk_distance
    out_filepath = args.out_filepath
    image_key = args.image_key
    apply = args.apply
    pivot = args.pivot
    verbose = args.verbose

    from squirrel.workflows.transformation import apply_from_z_chunks
    apply_from_z_chunks(
        image_filepath,
        transforms,
        chunk_distance,
        out_filepath=out_filepath,
        image_key=image_key,
        apply=apply,
        pivot=pivot,
        verbose=verbose
    )


def apply_rotation_and_scale():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Applies an average of multiple affine transforms to a volume.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('image_filepath', type=str,
                        help='Input filepath for the moving volume (nii or h5)')
    parser.add_argument('transform_filepath', type=str,
                        help='Json file containing the transformation')
    parser.add_argument('stack_item_distance', type=int,
                        help='The distance in pixels of the sub-stacks')
    parser.add_argument('-o', '--out_filepath', type=str, default=None,
                        help='Output filepath for the result file (only h5 for now)')
    parser.add_argument('--image_key', type=str, default='data',
                        help='Internal path of the moving input; default="data"; used if moving_filepath is h5 file')
    parser.add_argument('--apply', type=str, nargs='+', default=('rotates', 'scale', 'translate'),
                        help='Applies the defined transform components to the volume')
    parser.add_argument('--pivot', type=float, nargs=3, default=(0., 0., 0.),
                        help='Point around which scaling is centered. Caution: This does not work for rotation (yet?)')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    image_filepath = args.image_filepath
    transform_filepath = args.transform_filepath
    stack_item_distance = args.stack_item_distance
    out_filepath = args.out_filepath
    image_key = args.image_key
    apply = args.apply
    pivot = args.pivot
    verbose = args.verbose

    from squirrel.workflows.transformation import apply_rotation_and_scale_from_transform_stack
    apply_rotation_and_scale_from_transform_stack(
        image_filepath,
        transform_filepath,
        stack_item_distance,
        out_filepath=out_filepath,
        image_key=image_key,
        apply=apply,
        pivot=pivot,
        verbose=verbose
    )


def apply_auto_pad():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Computes auto-padding information for a stack of transformations.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('transform_filepath', type=str,
                        help='Json file containing the transformation')
    parser.add_argument('out_filepath', type=str,
                        help='Json file to which the result is saved')
    parser.add_argument('--image_stack_path', type=str, default=None,
                        help='file/dir-path of the image stack corresponding to transformations')
    parser.add_argument('--key', type=str, default='data',
                        help='Internal path of the moving input; default="data"; used if moving_filepath is h5 file')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='Used to glob tif files from a tif stack folder; default="*.tif"')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    transform_filepath = args.transform_filepath
    out_filepath = args.out_filepath
    image_stack_path = args.image_stack_path
    key = args.key
    pattern = args.pattern
    verbose = args.verbose

    from squirrel.workflows.transformation import apply_auto_pad_workflow
    apply_auto_pad_workflow(
        transform_filepath,
        out_filepath,
        image_stack_path=image_stack_path,
        key=key,
        pattern=pattern,
        verbose=verbose
    )


def get_affine_transforms_substack():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Extracts a substack of affine transforms from a stack of transformations.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('transform_filepath', type=str,
                        help='Json file containing the transformation')
    parser.add_argument('out_filepath', type=str,
                        help='Json file to which the result is saved')
    parser.add_argument('--z_range', type=int, nargs=2, default=None,
                        help='The z-range of the substack to extract; default=None (the entire stack)')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    transform_filepath = args.transform_filepath
    out_filepath = args.out_filepath
    z_range = args.z_range
    verbose = args.verbose

    from squirrel.workflows.transformation import get_affine_transforms_substack_workflow
    get_affine_transforms_substack_workflow(
        transform_filepath,
        out_filepath,
        z_range=z_range,
        verbose=verbose
    )


if __name__ == '__main__':
    apply_stack_alignment()
