import os.path

import numpy as np


def decompose_affine(
        transform,
        out_folder=None,
        shear_to_translation_pivot=None,
        pivot=None,
        verbose=False
):
    # from ..library.transformation import (
    #     load_transform_matrix,
    #     extract_approximate_rotation_affine,
    #     decompose_3d_transform,
    #     validate_and_reshape_matrix
    # )
    # from ..library.elastix import save_transforms

    # if type(transform) == str:
    #     transform = load_transform_matrix(transform)
    #
    # if verbose:
    #     print(f'transform = {transform}')

    # transform = validate_and_reshape_matrix(transform, 3)

    from ..library.affine_matrices import AffineMatrix
    if type(transform) == str:
        transform = AffineMatrix(filepath=transform)
    else:
        transform = AffineMatrix(parameters=transform)

    if verbose:
        print(f'transform.get_matrix() = {transform.get_matrix()}')

    if shear_to_translation_pivot is not None:
        tpivot = AffineMatrix(np.array(
            [
                [1, 0, 0, shear_to_translation_pivot[0]],
                [0, 1, 0, shear_to_translation_pivot[1]],
                [0, 0, 1, shear_to_translation_pivot[2]]
            ]
        ).flatten())
        tpivot_ = AffineMatrix(np.array(
            [
                [1, 0, 0, -shear_to_translation_pivot[0]],
                [0, 1, 0, -shear_to_translation_pivot[1]],
                [0, 0, 1, -shear_to_translation_pivot[2]]
            ]
        ).flatten())
        transform = transform.dot(tpivot)
    if verbose:
        print(f'transform = {transform}')
    decomposition = transform.decompose()
    # decomposition = decompose_3d_transform(transform, verbose=verbose)
    if verbose:
        print(f'decomposition = {decomposition}')

    # if shear_to_translation_pivot is not None:
    #     import math
    #     shear = decomposition[3]
    #     translation = decomposition[0]
    #     translation[0] += shear_to_translation_pivot[0] * math.atan(shear[0])
    #     translation[1] += shear_to_translation_pivot[1] * math.atan(shear[1])
    #     translation[2] += shear_to_translation_pivot[2] * math.atan(shear[2])

    # from ..library.transformation import (
    #     setup_translation_matrix,
    #     setup_scale_matrix,
    #     setup_shear_matrix
    # )
    #
    # translation = decomposition[0]
    # translation = setup_translation_matrix(translation)
    # if shear_to_translation_pivot is not None:
    #     translation = np.dot(translation, tpivot_)
    # scale = decomposition[2]
    # scale = setup_scale_matrix(scale)
    # rotation = np.concatenate([decomposition[1], np.swapaxes([[0., 0., 0.]], 0, 1)], axis=1)
    # shear = decomposition[3]
    # shear = setup_shear_matrix(shear)

    translation, rotation, scale, shear = decomposition
    if shear_to_translation_pivot:
        translation = translation.dot(tpivot_)

    if verbose:
        print(f'translation = {translation.get_matrix()}')
        print(f'rotation = {rotation.get_matrix()}')
        print(f'scale = {scale.get_matrix()}')
        print(f'shear = {shear.get_matrix()}')

    if out_folder is not None:
        import os
        translation.to_file(os.path.join(out_folder, 'translation.json'))
        rotation.to_file(os.path.join(out_folder, 'rotation.json'))
        scale.to_file(os.path.join(out_folder, 'scale.json'))
        shear.to_file(os.path.join(out_folder, 'shear.json'))
        # save_transforms(translation, os.path.join(out_folder, 'translation.json'), param_order='M', save_order='C', ndim=3, verbose=verbose)
        # save_transforms(rotation, os.path.join(out_folder, 'rotation.json'), param_order='M', save_order='C', ndim=3, verbose=verbose)
        # save_transforms(scale, os.path.join(out_folder, 'scale.json'), param_order='M', save_order='C', ndim=3, verbose=verbose)
        # save_transforms(shear, os.path.join(out_folder, 'shear.json'), param_order='M', save_order='C', ndim=3, verbose=verbose)
    return decomposition


def apply_affine(
        image,
        transform,
        out_filepath=None,
        image_key='data',
        no_offset_to_center=False,
        # pivot=None,
        apply='all',  # Can be ['all' | 'rotation']
        scale_canvas=False,
        verbose=False
):

    if verbose:
        print(f'image = {image if type(image) == str else image.shape}')
        print(f'transform = {transform}')
        print(f'out_filepath = {out_filepath}')
        print(f'image_key = {image_key}')

    from ..library.transformation import apply_affine_transform
    # from ..library.io import load_data, write_h5_container
    from ..library.io import load_data_handle, write_h5_container

    if type(image) == str:
        # image = load_data(image, key=image_key)
        image, _ = load_data_handle(image, key=image_key)[:]

    if type(transform) == str:
        # from ..library.transformation import load_transform_matrix
        from ..library.affine_matrices import AffineMatrix
        # transform = load_transform_matrix(transform)
        transform = AffineMatrix(filepath=transform)

    result = apply_affine_transform(
        image,
        transform,
        no_offset_to_center=no_offset_to_center,
        # pivot=pivot,
        apply=apply,
        scale_canvas=scale_canvas,
        verbose=verbose
    )

    write_h5_container(out_filepath, result, key='data')

    return result


def apply_sequential_affine(
        image,
        transforms,
        out_filepath=None,
        image_key='data',
        no_offset_to_center=False,
        pivot=None,
        verbose=False
):

    from ..library.affine_matrices import AffineMatrix

    transform = None
    for t in transforms:
        if transform is None:
            transform = AffineMatrix(filepath=t)
        else:
            transform = transform.dot(AffineMatrix(filepath=t))

    transform.set_pivot(pivot)

    transform.to_file(
        os.path.join(
            os.path.split(out_filepath)[0],
            os.path.splitext(os.path.split(out_filepath)[1])[0] + '.json'
        )
    )

    # from ..library.transformation import load_transform_matrix, validate_and_reshape_matrix
    # if len(transforms) == 1:
    #     transforms = load_transform_matrix(transforms[0])
    #     transform = validate_and_reshape_matrix(transforms[0], 3)
    #     for t in transforms[1:]:
    #         transform = np.dot(transform, validate_and_reshape_matrix(t, 3))
    #
    # else:
    #     transform = validate_and_reshape_matrix(load_transform_matrix(transforms[0]), 3)
    #     for t in transforms[1:]:
    #         t = validate_and_reshape_matrix(load_transform_matrix(t), 3)
    #         transform = np.dot(transform, t)
    #
    # from ..library.elastix import save_transforms
    # import os
    # save_transforms(
    #     transform[:3, :],
    #     os.path.join(
    #         os.path.split(out_filepath)[0],
    #         os.path.splitext(os.path.split(out_filepath)[1])[0] + '.json'
    #     ),
    #     param_order='M',
    #     save_order='C',
    #     ndim=3,
    #     verbose=verbose
    # )

    apply_affine(
        image,
        transform,
        out_filepath=out_filepath,
        image_key=image_key,
        no_offset_to_center=no_offset_to_center,
        verbose=verbose
    )


def apply_average_affine(
        image,
        transform,
        out_filepath=None,
        image_key='data',
        method='mean',
        verbose=False
):

    if type(transform) == str:
        import json
        with open(transform, mode='r') as f:
            transform = np.array(json.load(f))

    if method == 'mean':
        transform = np.mean(transform, axis=0)
    elif method == 'median':
        transform = np.median(transform, axis=0)
    else:
        raise ValueError(f'Invalid averaging method: {method}')

    return apply_affine(
        image,
        transform,
        out_filepath=out_filepath,
        image_key=image_key,
        verbose=verbose
    )


def apply_from_z_chunks(
        image,
        transforms,
        chunk_distance,
        out_filepath=None,
        image_key='data',
        apply=('rotate', 'scale', 'translate', 'shear'),  # also: ('all', 'scale-xy/-z')
        pivot=(0., 0., 0.),
        verbose=False
):

    from ..library.transformation import (
        decompose_3d_transform,
        validate_and_reshape_matrix,
        setup_translation_matrix,
        setup_rotation_matrix,
        setup_scale_matrix,
        setup_shear_matrix
    )
    from ..library.elastix import get_affine_rotation_parameters, save_transforms

    if type(transforms) == str:
        import json
        with open(transforms, mode='r') as f:
            transforms = json.load(f)

    def _matrices_from_affine(transforms):

        translations = []
        rotations = []
        scales = []
        shears = []

        for idx, t in enumerate(transforms):
            t = validate_and_reshape_matrix(t, 3)
            translation, rotation, scale, shear = decompose_3d_transform(t)
            translations.append(translation)
            rotations.append(rotation)
            scales.append(scale)
            shears.append(shear)

            print(f'translation = {translation}')
            print(f'rotation = {rotation}')
            print(f'scale = {scale}')
            print(f'shear = {shear}')

        # Rotation and translation are just averaged from each chunk
        rotation = setup_rotation_matrix(np.mean(rotations, axis=0))
        translation = setup_translation_matrix(np.mean(translations, axis=0))

        # z-scale is handled individually from the z-translation of first and last chunk
        stack_height = chunk_distance * len(translations)
        z_scale = stack_height / (stack_height + translations[0][0] - translations[-1][0])
        z_scale = setup_scale_matrix([z_scale, z_scale, z_scale])
        if verbose:
            print(f'z_scale = {z_scale}')

        # x- and y- scales are averaged
        xy_scale = setup_scale_matrix(np.mean(scales, axis=0))
        xy_scale[0, 0] = 0.

        # The xy-shearing is defined by the x- and y- translations
        # TODO! but ignoring for now

        return rotation, translation, z_scale, xy_scale

    def _matrices_from_similarity(transforms):

        translations = []
        rotations = []
        scales = []

        for idx, t in enumerate(transforms):
            rotations.append(t[:3])
            translations.append(t[3:6][::-1])
            scales.append(t[6])

        # Rotation and translation are just averaged from each chunk
        rotation = np.mean(rotations, axis=0)
        translation = np.mean(translations, axis=0)

        # z-scale is handled individually from the z-translation of first and last chunk
        stack_height = chunk_distance * len(translations)
        z_scale = stack_height / (stack_height + translations[0][0] - translations[-1][0])

        # x- and y- scales are averaged
        xy_scale = np.mean(scales, axis=0)

        # Set up the matrices
        rotation = save_transforms(
            list(get_affine_rotation_parameters(rotation)) + [0., 0., 0.],
            None, param_order='elastix', save_order='M', ndim=3, verbose=verbose
        )
        translation = setup_translation_matrix(translation)
        xy_scale = setup_scale_matrix([1., xy_scale, xy_scale])
        z_scale = setup_scale_matrix([z_scale, z_scale, z_scale])

        return rotation, translation, z_scale, xy_scale

    if len(transforms[0]) == 12:
        rotation, translation, z_scale, xy_scale = _matrices_from_affine(transforms)
    elif len(transforms[0]) == 7:
        rotation, translation, z_scale, xy_scale = _matrices_from_similarity(transforms)
    else:
        raise ValueError(f'transform has invalid length: {len(transforms[0])}')

    if verbose:
        print(f'rotation = {rotation}')
        print(f'translation = {translation}')
        print(f'z_scale = {z_scale}')
        print(f'xy_scale = {xy_scale}')

    rotation = validate_and_reshape_matrix(rotation, 3)
    translation = validate_and_reshape_matrix(translation, 3)
    z_scale = validate_and_reshape_matrix(z_scale, 3)
    xy_scale = validate_and_reshape_matrix(xy_scale, 3)

    if verbose:
        print(f'rotation = {rotation}')
        print(f'translation = {translation}')
        print(f'z_scale = {z_scale}')
        print(f'xy_scale = {xy_scale}')

    transform = rotation if 'rotate' in apply else np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    assert transform[0, 3] == 0.
    assert transform[1, 3] == 0.
    assert transform[2, 3] == 0.

    if 'translate' in apply:
        if verbose:
            print(f'transform = {transform}')
            print(f'translation = {translation}')
        transform = np.dot(transform, translation)
    if 'scale-z' in apply or 'scale' in apply:
        transform = np.dot(transform, z_scale)
    if 'scale-xy' in apply or 'scale' in apply:
        transform = np.dot(transform, xy_scale)

    from ..library.elastix import save_transforms
    import os
    save_transforms(
        transform[:3, :],
        os.path.join(
            os.path.split(out_filepath)[0],
            os.path.splitext(os.path.split(out_filepath)[1])[0] + '.json'
        ),
        param_order='M', save_order='C', ndim=3, verbose=verbose
    )

    return apply_affine(
        image,
        transform,
        out_filepath=out_filepath,
        image_key=image_key,
        no_offset_to_center=False,
        pivot=pivot,
        verbose=verbose
    )


def apply_rotation_and_scale_from_transform_stack(
        image,
        transform_stack,
        stack_item_distance,  # The distance in pixels
        out_filepath=None,
        image_key='data',
        apply=('rotate', 'scale', 'translate'),
        pivot=(0., 0., 0.),
        verbose=False
):
    """
    Note that apply=('scale') applies the z-scale to all dimensions.
    """

    if verbose:
        print(f'apply = {apply}')

    from ..library.transformation import extract_approximate_rotation_affine
    from ..library.transformation import validate_and_reshape_matrix

    if type(transform_stack) == str:
        import json
        with open(transform_stack, mode='r') as f:
            transform_stack = json.load(f)

    if verbose:
        print(f'transform_stack = {transform_stack}')
    rotation = []
    z_displacements = []
    y_displacements = []
    x_displacements = []

    for idx, t in enumerate(transform_stack):
        t = validate_and_reshape_matrix(t, 3)
        rotation.append(extract_approximate_rotation_affine(t, 0))
        z_displacements.append(t[0, 3])
        y_displacements.append(t[1, 3])
        x_displacements.append(t[2, 3])
        if verbose:
            print(f't = {t}')
    rotation = np.mean(rotation, axis=0)
    if verbose:
        print(f'rotation = {rotation}')
        print(f'x_displacements = {x_displacements}')
        print(f'y_displacements = {y_displacements}')
        print(f'z_displacements = {z_displacements}')

    stack_height = stack_item_distance * len(z_displacements)
    z_scale = stack_height / (stack_height + z_displacements[0] - z_displacements[-1])
    if verbose:
        print(f'z_scale = {z_scale}')

    z_displacement = np.mean(z_displacements)
    y_displacement = np.mean(y_displacements)
    x_displacement = np.mean(x_displacements)

    transform = rotation if 'rotate' in apply else np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
    assert transform[0, 3] == 0.
    assert transform[1, 3] == 0.
    assert transform[2, 3] == 0.
    if 'scale-z' in apply or 'scale' in apply:
        print('scale-z')
        transform[0, 0] = z_scale
        transform[0, 3] += pivot[0] - pivot[0] * z_scale
    if 'scale-y' in apply or 'scale' in apply:
        print('scale-y')
        transform[1, 1] = z_scale
        transform[1, 3] += pivot[1] - pivot[1] * z_scale
    if 'scale-x' in apply or 'scale' in apply:
        print('scale-x')
        transform[2, 2] = z_scale
        transform[2, 3] += pivot[2] - pivot[2] * z_scale
    if 'translate-z' in apply or 'translate' in apply:
        print('translate-z')
        transform[0, 3] += z_displacement
    if 'translate-y' in apply or 'translate' in apply:
        print('translate-y')
        transform[1, 3] += y_displacement
    if 'translate-x' in apply or 'translate' in apply:
        print('translate-x')
        transform[2, 3] += x_displacement

    from ..library.elastix import save_transforms
    import os
    save_transforms(
        transform[:3, :], os.path.join(os.path.split(out_filepath)[0], 'transform.json'),
        param_order='M', save_order='C', ndim=3, verbose=verbose
    )

    return apply_affine(
        image,
        transform,
        out_filepath=out_filepath,
        image_key=image_key,
        no_offset_to_center=True,
        pivot=None,
        verbose=verbose
    )


def apply_stack_alignment_on_volume_workflow(
        stack,
        transform_filepath,
        out_filepath,
        key='data',
        pattern='*.tif',
        no_adding_of_transforms=False,
        auto_pad=False,
        z_range=None,
        stack_shape=None,
        n_workers=1,
        quiet=False,
        verbose=False,
):

    from squirrel.library.io import load_data_handle, write_h5_container
    from squirrel.library.transformation import apply_stack_alignment
    from squirrel.library.affine_matrices import AffineMatrix, AffineStack

    transforms = AffineStack(filepath=transform_filepath)
    if verbose:
        print(f'is_sequenced = {transforms.is_sequenced}')
    if not transforms.is_sequenced and not no_adding_of_transforms:
        if verbose:
            print(f'Sequencing stack!')
        transforms = transforms.get_sequenced_stack()

    if stack_shape is None:
        stack_h, stack_shape = load_data_handle(stack, key=key, pattern=pattern)
    else:
        assert not auto_pad, "Don't supply a stack shape if auto padding will be performed!"
        stack_h, _ = load_data_handle(stack, key=key, pattern=pattern)
    if transforms.exists_meta('stack_shape'):
        stack_shape = transforms.get_meta('stack_shape')
    stack_len = stack_shape[0]
    if z_range is not None:
        stack_len = z_range[1] - z_range[0]
    if verbose:
        print(f'transforms = {transforms["M", :]}')
    if auto_pad:
        from squirrel.library.image import get_bounds_of_stack, apply_auto_pad
        stack_bounds = get_bounds_of_stack(stack_h, stack_shape, return_ints=True, z_range=z_range)
        if verbose:
            print(f'stack_bounds = {stack_bounds}')
        transforms, stack_shape = apply_auto_pad(
            transforms, [stack_len, *stack_shape[1:]], stack_bounds, extra_padding=16
        )
        if verbose:
            print(f'transforms = {transforms["M", :]}')
            print(f'stack_shape = {stack_shape}')

    result_volume = apply_stack_alignment(
        stack_h,
        stack_shape,
        transforms,
        no_adding_of_transforms=True,
        z_range=z_range,
        n_workers=n_workers,
        quiet=quiet,
        verbose=verbose
    )

    write_h5_container(out_filepath, result_volume)


def dot_product_on_affines_workflow(
        transform_filepaths,
        out_filepath,
        inverse=(0, 0),
        keep_meta=None,
        verbose=False
):

    if verbose:
        print(f'transform_filepaths = {transform_filepaths}')
        print(f'out_filepath = {out_filepath}')
        print(f'inverse = {inverse}')
        print(f'keep_meta = {keep_meta}')

    from ..library.affine_matrices import AffineStack

    transforms = [
        AffineStack(filepath=transform_filepaths[0]),
        AffineStack(filepath=transform_filepaths[1])
    ]
    if transforms[0].is_sequenced or transforms[1].is_sequenced:
        if not transforms[0].is_sequenced:
            transforms[0] = transforms[0].get_sequenced_stack()
        if not transforms[1].is_sequenced:
            transforms[1] = transforms[1].get_sequenced_stack()

    if inverse[0]:
        transforms[0] = -transforms[0]
    if inverse[1]:
        transforms[1] = -transforms[1]
    out_transforms = transforms[0] * transforms[1]
    if keep_meta is not None:
        out_transforms.set_meta(data=transforms[keep_meta].get_meta())

    out_transforms.to_file(out_filepath)


def scale_sequential_affines_workflow(
        transform_filepath,
        out_filepath,
        scale,
        xy_pivot=(0., 0.),
        verbose=False
):
    from ..library.transformation import load_transform_matrices, scale_sequential_affines
    from ..library.elastix import save_transforms
    transforms = np.array(load_transform_matrices(transform_filepath, validate=True, ndim=2))

    if verbose:
        print(f'scale = {scale}')
        print(f'transforms.shape = {transforms.shape}')

    transforms = scale_sequential_affines(transforms, scale, xy_pivot)

    if verbose:
        print(f'transforms.shape = {transforms.shape}')

    # Prepare for saving
    transforms = [save_transforms(x, None, param_order='M', save_order='C', ndim=2)[:6].tolist() for x in transforms]

    import json
    with open(out_filepath, mode='w') as f:
        json.dump(transforms, f, indent=2)


def sequence_affine_stack_workflow(
        transform_filepath,
        out_filepath,
        verbose=False
):

    if os.path.exists(out_filepath):
        print(f'Target file exists: {out_filepath}\nSkipping apply affine sequence workflow ...')
        return None

    from squirrel.library.affine_matrices import AffineStack

    transforms = AffineStack(filepath=transform_filepath)
    transforms = transforms.get_sequenced_stack()
    transforms.to_file(out_filepath)

    # import json
    # with open(transform_filepath, mode='r') as f:
    #     transforms = json.load(f)
    #
    # from squirrel.library.transformation import sequence_affine_stack
    # result_transforms = sequence_affine_stack(transforms, verbose=verbose)
    #
    # with open(out_filepath, mode='w') as f:
    #     json.dump(result_transforms, f, indent=2)


def smooth_affine_sequence_workflow(
        transform_filepath,
        out_filepath,
        sigma,
        components=None,
        mode='reflect',
        verbose=False
):

    if components is not None:
        raise NotImplementedError

    from ..library.affine_matrices import AffineStack

    transforms = AffineStack(filepath=transform_filepath)
    transforms = transforms.get_smoothed_stack(sigma, mode=mode)
    transforms.to_file(out_filepath)


def inverse_of_sequence_workflow(
        transform_filepath,
        out_filepath,
        verbose=False
):
    from ..library.elastix import save_transforms
    from ..library.transformation import load_transform_matrices
    from ..library.linalg import inverse_of_sequence

    transforms = np.array(load_transform_matrices(transform_filepath, validate=True, ndim=2))

    if verbose:
        print(f'transforms_a.shape = {transforms.shape}')
    # assert transforms_a.shape == transforms_b.shape, \
    #     f'Shapes of the transform sequences have to match: {transforms_a.shape} != {transforms_b.shape}'

    result_transforms = inverse_of_sequence(transforms)

    # Prepare for saving
    transforms = [
        save_transforms(x, None, param_order='M', save_order='C', ndim=2)[:6].tolist()
        for x in result_transforms
    ]

    import json
    with open(out_filepath, mode='w') as f:
        json.dump(transforms, f, indent=2)


def add_translational_drift_workflow(
        transform_filepath,
        out_filepath,
        drift,
        is_serialized=False,
        verbose=False
):

    if verbose:
        print(f'transform_filepath = {transform_filepath}')
        print(f'out_filepath = {out_filepath}')
        print(f'drift = {drift}')
        print(f'is_serialized = {is_serialized}')

    from ..library.affine_matrices import AffineStack

    transforms = AffineStack(filepath=transform_filepath)

    if transforms.is_sequenced:
        transforms.add_to_translations([np.array(drift) * x for x in range(len(transforms))])
    else:
        transforms.add_to_translations(drift)

    transforms.to_file(out_filepath)

    # from ..library.elastix import save_transforms
    # from ..library.transformation import load_transform_matrices, save_transformation_matrices
    # transforms, sequenced = load_transform_matrices(transform_filepath, validate=True, ndim=2)
    # transforms = np.array(transforms)
    #
    # if sequenced is not None:
    #     is_serialized = sequenced
    #
    # if is_serialized:
    #     drift = [np.array(drift) * x for x in range(len(transforms))]
    # else:
    #     drift = [drift] * len(transforms)
    #
    # for idx, transform in enumerate(transforms):
    #     transforms[idx][:2, 2] += drift[idx]
    #
    # # Prepare for saving
    # transforms = [
    #     save_transforms(x, None, param_order='M', save_order='C', ndim=2)[:6].tolist()
    #     for x in transforms
    # ]
    #
    # save_transformation_matrices(out_filepath, transforms, sequenced=sequenced)
    #
    # # import json
    # # with open(out_filepath, mode='w') as f:
    # #     json.dump(transforms, f, indent=2)


def modify_step_in_sequence_workflow(transform_filepath, out_filepath, idx, affine, replace=False, return_result=False, verbose=False):

    if verbose:
        print(f'transform_filepath = {transform_filepath}')
        print(f'idx = {idx}')
        print(f'affine = {affine}')
        print(f'replace = {replace}')

    # from ..library.linalg import modify_step_in_sequence
    # from ..library.transformation import load_transform_matrices, save_transformation_matrices
    # from ..library.elastix import save_transforms
    #
    # transforms, sequenced = load_transform_matrices(transform_filepath, validate=True, ndim=2)
    # transforms = modify_step_in_sequence(transforms, idx, affine, replace=replace)
    # save_transformation_matrices(
    #     out_filepath,
    #     save_transforms(transforms, None, param_order='M', save_order='C', ndim=2),
    #     sequenced=sequenced
    # )

    from squirrel.library.affine_matrices import AffineStack, AffineMatrix
    if not isinstance(transform_filepath, AffineStack):
        transforms = AffineStack(filepath=transform_filepath)
    else:
        transforms = transform_filepath
    if replace:
        assert not transforms.is_sequenced
        transforms[idx] = AffineMatrix(parameters=affine)
    if transforms.is_sequenced:
        for tidx, transform in enumerate(transforms[idx:]):
            transforms[idx + tidx] = transform * AffineMatrix(parameters=affine)
    else:
        transforms[idx] = transforms[idx] * AffineMatrix(parameters=affine)

    if return_result:
        return transforms
    transforms.to_file(out_filepath)


def create_affine_sequence_workflow(
        out_filepath,
        length,
        from_transform_file=None,
        sequenced=False,
        verbose=False
):

    if verbose:
        print(f'out_filepath = {out_filepath}')
        print(f'length = {length}')
        print(f'from_transform_file = {from_transform_file}')
        print(f'sequenced = {sequenced}')

    from squirrel.library.affine_matrices import AffineStack, AffineMatrix

    if from_transform_file is None:
        matrix = AffineMatrix(parameters=np.eye(3)[:2].flatten())
    else:
        matrix = AffineMatrix(filepath=from_transform_file)

    transforms = AffineStack(stack=[matrix] * length, is_sequenced=sequenced)
    transforms.to_file(out_filepath)


def apply_auto_pad_workflow(
        transform_filepath, out_filepath,
        image_stack_path=None, key='data', pattern='*.tif',
        verbose=False
):

    if verbose:
        print(f'transform_filepath = {transform_filepath}')
        print(f'out_filepath = {out_filepath}')

    from squirrel.library.image import apply_auto_pad
    from squirrel.library.affine_matrices import AffineStack

    transforms = AffineStack(filepath=transform_filepath)
    if not transforms.is_sequenced:
        transforms = transforms.get_sequenced_stack()
    if not transforms.exists_meta('bounds'):
        from squirrel.library.image import get_bounds_of_stack, apply_auto_pad
        from squirrel.library.io import load_data_handle
        assert image_stack_path is not None, 'A stack needs to be supplied if no bounds information is found in the transformation meta data'
        stack_h, stack_shape = load_data_handle(image_stack_path, key=key, pattern=pattern)
        stack_bounds = get_bounds_of_stack(stack_h, stack_shape, return_ints=True, z_range=None)
    else:
        stack_bounds = transforms.get_meta('bounds')

    transforms, stack_shape = apply_auto_pad(
        transforms, [len(transforms), 0, 0], stack_bounds, extra_padding=16
    )
    transforms.set_meta('stack_shape', stack_shape)
    transforms.to_file(out_filepath)


def crop_transform_sequence_workflow(transform_filepath, out_filepath, z_range, verbose=False):

    if verbose:
        print(f'transform_filepath = {transform_filepath}')
        print(f'out_filepath = {out_filepath}')
        print(f'z_range = {z_range}')

    from squirrel.library.affine_matrices import AffineStack
    stack = AffineStack(filepath=transform_filepath)
    out_stack = stack.new_stack_with_same_meta(stack[z_range[0]: z_range[1]])

    out_stack.to_file(out_filepath)


def get_affine_transforms_substack_workflow(transform_filepath, out_filepath, z_range, verbose=False):

    if verbose:
        print(f'transform_filepath = {transform_filepath}')
        print(f'out_filepath = {out_filepath}')
        print(f'z_range = {z_range}')

    from squirrel.library.affine_matrices import AffineStack

    stack = AffineStack(filepath=transform_filepath)

    import numpy as np
    substack = stack.get_substack(np.s_[z_range[0]: z_range[1]])

    substack.to_file(out_filepath)


if __name__ == '__main__':

    # from squirrel.library.affine_matrices import AffineStack, AffineMatrix
    #
    # affines = AffineStack(stack = [[1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 1, 0]], is_sequenced=True)
    #
    # print(affines['C', :])
    #
    # modified_affines = modify_step_in_sequence_workflow(affines, None, 1, [1, 0, 10, 0, 1, 5], return_result=True)
    #
    # print(modified_affines['C', :])

    create_affine_sequence_workflow('test.json', 10, None, False)
