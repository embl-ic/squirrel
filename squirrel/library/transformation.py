import math

import numpy as np


def setup_translation_matrix(translation_zyx, ndim=3):

    if ndim == 3:

        return np.array(
            [
                [1., 0., 0., translation_zyx[0]],
                [0., 1., 0., translation_zyx[1]],
                [0., 0., 1., translation_zyx[2]]
            ]
        )

    if ndim == 2:

        return np.array(
            [
                [1., 0., translation_zyx[0]],
                [0., 1., translation_zyx[1]]
            ]
        )

    raise ValueError(f'Invalid number of dimensions = {ndim}')


def setup_rotation_matrix(rotation):
    return np.concatenate((rotation, np.swapaxes([[0., 0., 0.]], 0, 1)), axis=1)


def setup_2d_rotation_matrix_from_angle(angle):
    return np.array([
        [math.cos(angle), math.sin(angle), 0.],
        [-math.sin(angle), math.cos(angle), 0.]
    ])


def setup_scale_matrix(scale_zyx, ndim=3):

    import numbers

    assert len(scale_zyx) == ndim
    assert isinstance(scale_zyx[0], numbers.Number)

    if ndim == 3:
        return np.array(
            [
                [scale_zyx[0], 0., 0., 0.],
                [0., scale_zyx[1], 0., 0.],
                [0., 0., scale_zyx[2], 0.]
            ]
        )

    if ndim == 2:
        return np.array(
            [
                [scale_zyx[0], 0., 0.],
                [0., scale_zyx[1], 0.]
            ]
        )

    raise ValueError(f'Invalid number of dimensions = {ndim}')


def setup_shear_matrix(shear_zyx, ndim=3):

    if ndim == 3:
        return np.array(
            [
                [1., shear_zyx[0], shear_zyx[1], 0.],
                [0., 1., shear_zyx[2], 0.],
                [0., 0., 1., 0.]
            ]
        )

    if ndim == 2:
        return np.array(
            [
                [1., shear_zyx[0], 0.],
                [0., 1., 0.]
            ]
        )
    raise ValueError(f'Invalid number of dimensions = {ndim}')


def extract_approximate_rotation_affine(transform, coerce_affine_dimension):
    # print(transform)

    from copy import deepcopy
    new_transform = deepcopy(transform)

    # for c in range(3):
    #     sq_sum = 0
    #     for r in range(3):
    #         sq_sum += new_transform[r, c] ** 2
    #     s = 1. / math.sqrt(sq_sum)
    #     for r in range(3):
    #         new_transform[r, c] *= s

    x = new_transform[:3, 0]
    y = new_transform[:3, 1]
    z = new_transform[:3, 2]

    if coerce_affine_dimension == 0:
        x = np.cross(y, z)
        z = np.cross(x, y)
    if coerce_affine_dimension == 1:
        y = np.cross(z, x)
        x = np.cross(y, z)
    if coerce_affine_dimension == 2:
        z = np.cross(x, y)
        y = np.cross(z, x)

    new_transform[:3, 0] = x
    new_transform[:3, 1] = y
    new_transform[:3, 2] = z

    new_transform[0, 0] = 1.0
    new_transform[1, 1] = 1.0
    new_transform[2, 2] = 1.0

    new_transform[:3, 3] = 0.0

    return new_transform


def apply_affine_transform(
        x,
        transform,
        fill_mode='nearest',
        cval=0.,
        order=3,
        pivot=None,
        no_offset_to_center=False,
        apply='all',
        scale_canvas=False,
        verbose=False,
):

    from squirrel.library.affine_matrices import AffineMatrix

    x = np.asarray(x)

    if not isinstance(transform, AffineMatrix):
        transform = AffineMatrix.from_array(transform)
    else:
        transform = transform.copy()

    if transform.ndim != x.ndim:
        raise ValueError(f"Transform is {transform.ndim}D, but input image is {x.ndim}D.")
    if apply == 'rotation':
        raise NotImplementedError
    if apply != 'all':
        raise ValueError(f"Invalid apply mode: {apply!r}.")

    if pivot is not None:
        transform = transform.with_pivot(pivot)

    elif not no_offset_to_center:
        transform = transform.with_pivot(np.asarray(x.shape, dtype=float) / 2)

    scipy_matrix, scipy_offset = transform.as_scipy_affine()

    if verbose:
        print(f'x.ndim = {x.ndim}')
        print(f'x.shape = {x.shape}')
        print(f'transform = {transform}')
        print(f'scipy_matrix =\n{scipy_matrix}')
        print(f'scipy_offset = {scipy_offset}')

    import scipy.ndimage as ndi

    result = ndi.affine_transform(
        x,
        matrix=scipy_matrix,
        offset=scipy_offset,
        order=order,
        mode=fill_mode,
        cval=cval,
    )

    if scale_canvas:
        if not no_offset_to_center:
            raise ValueError("scale_canvas requires no_offset_to_center=True.")

        if not np.allclose(transform.pivot, np.zeros(transform.ndim)):
            raise ValueError("scale_canvas requires a zero pivot.")

        _, _, scale, _ = transform.decompose()
        scale_factors = np.diag(scale.linear)

        crop = np.floor(np.asarray(x.shape) / scale_factors).astype(int)

        result = result[tuple(slice(0, size) for size in crop)]

    return result


def apply_stack_alignment_slice(
        stack_h,
        stack_shape,
        transform,
        idx,
        n_slices=None,
        quiet=False,
        verbose=False
):

    from squirrel.library.io import get_reshaped_data

    if not quiet:
        print(f'idx = {idx} / {n_slices}')

    z_slice = get_reshaped_data(stack_h, idx, stack_shape[1:])
    return apply_affine_transform(
        z_slice, transform,
        fill_mode='constant',
        cval=0,
        verbose=verbose
    )[:stack_shape[1], :stack_shape[2]]


def apply_stack_alignment(
        stack_h,
        stack_shape,
        transform_stack,
        no_adding_of_transforms=False,
        z_range=None,
        n_workers=1,
        quiet=False,
        verbose=False
):

    if verbose:
        print(f'transform_stack.is_sequenced = {transform_stack.sequenced}')
    if not transform_stack.sequenced and not no_adding_of_transforms:
        if verbose:
            print(f'sequencing stack!')
        transform_stack = transform_stack.to_sequenced()

    stack_size = np.ceil(np.array(stack_shape)).astype(int)

    result_volume = []

    from ..library.data import norm_z_range
    z_range = norm_z_range(z_range, stack_size[0])

    if n_workers == 1:

        for stack_idx, idx in enumerate(range(*z_range)):
            if verbose:
                print(f'stack_idx = {stack_idx}')
                print(f'len(transform_stack) = {len(transform_stack)}')
                print(f'transform_stack[stack_idx] = {transform_stack[stack_idx].get_matrix()}')

            result_volume.append(apply_stack_alignment_slice(
                stack_h,
                stack_size,
                transform_stack[stack_idx],
                idx,
                n_slices=z_range[1],
                quiet=quiet,
                verbose=verbose
            ))

    else:

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as tpe:
            tasks = [
                tpe.submit(
                    apply_stack_alignment_slice,
                    stack_h,
                    stack_size,
                    transform_stack[stack_idx],
                    idx,
                    z_range[1],
                    quiet,
                    verbose
                )
                for stack_idx, idx in enumerate(range(*z_range))
            ]
            result_volume = [task.result() for task in tasks]

    return np.array(result_volume)

