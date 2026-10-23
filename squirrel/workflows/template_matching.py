
import os
import numpy as np


def template_matching_stack_alignment_workflow(
        stack,
        out_filepath,
        template_roi,
        search_roi=None,
        resolution=(1., 1., 1.),
        key='data',
        pattern='*.tif',
        z_range=None,
        save_template=False,
        determine_bounds=False,
        n_threads=1,
        verbose=False
):

    if os.path.exists(out_filepath):
        print(f'Target file exists: {out_filepath}\nSkipping elastix stack alignment workflow ...')
        return

    if verbose:
        print(f'stack = {stack}')
        print(f'out_filepath = {out_filepath}')
        print(f'template_roi = {template_roi}')
        print(f'search_roi = {search_roi}')
        print(f'resolution = {resolution}')
        print(f'key = {key}')
        print(f'pattern = {pattern}')
        print(f'z_range = {z_range}')
        print(f'save_template = {save_template}')

    from ..library.io import load_data_handle, crop_roi
    from ..library.template_matching import match_template_on_stack_slice
    from ..library.affine_matrices import AffineStack
    from ..library.data import resolution_to_pixels, norm_z_range

    template_roi = np.array(template_roi)
    template_roi[[0, 2]] = resolution_to_pixels(template_roi[[0, 2]], resolution[2])
    template_roi[[1, 3]] = resolution_to_pixels(template_roi[[1, 3]], resolution[1])
    template_roi[4] = resolution_to_pixels(template_roi[4], resolution[0])
    template_roi = template_roi.tolist()
    if search_roi is not None:
        search_roi = np.array(search_roi)
        search_roi[[0, 2]] = resolution_to_pixels(search_roi[[0, 2]], resolution[2])
        search_roi[[1, 3]] = resolution_to_pixels(search_roi[[1, 3]], resolution[1])
        search_roi = search_roi.tolist()

    if verbose:
        print(f'After converting to pixels:')
        print(f'template_roi = {template_roi}')
        print(f'search_roi = {search_roi}')

    stack_h, stack_size = load_data_handle(stack, key=key, pattern=pattern)

    template = crop_roi(stack_h, template_roi)
    if verbose:
        print(f'stack_h.shape = {stack_h.shape}')
        print(f'template.shape = {template.shape}')

    if save_template:
        template_filepath = os.path.splitext(out_filepath)[0] + '.template.tif'
        from tifffile import imwrite
        imwrite(template_filepath, template)

    transforms = AffineStack(sequenced=True)
    bounds = []

    z_range = norm_z_range(z_range, stack_size[0])

    if n_threads == 1:

        for idx in range(*z_range):
            print(f'idx = {idx} / {z_range[1]}')
            this_transform, this_bounds = match_template_on_stack_slice(
                stack_h, idx, search_roi, template, determine_bounds=determine_bounds
            )
            transforms.append(this_transform)
            if determine_bounds:
                bounds.append(this_bounds)

    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_threads) as tpe:
            tasks = [
                tpe.submit(
                    match_template_on_stack_slice,
                    stack_h, idx, search_roi, template, determine_bounds=determine_bounds
                )
                for idx in range(*z_range)
            ]
            for idx, task in enumerate(tasks):
                print(f'idx = {idx} / {len(tasks) - 1}')
                this_transform, this_bounds = task.result()
                transforms.append(this_transform)
                if determine_bounds:
                    bounds.append(this_bounds)

    transforms.set_metadata('bounds', np.array(bounds))
    transforms.write(out_filepath)
