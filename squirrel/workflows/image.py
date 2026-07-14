
import os
import numpy as np


def filter_2d_workflow(
        input_filepath,
        out_filepath,
        filters=None,
        save_intermediates=False,
        verbose=False
):
    if verbose:
        print(f'input_filepath = {input_filepath}')
        print(f'out_filepath = {out_filepath}')
        print(f'filters = {filters}')

    from squirrel.library.filters import ImageFilter

    from squirrel.library.io import read_tif_slice, write_tif_slice
    img = read_tif_slice(input_filepath, return_filepath=False)

    imf = ImageFilter(img)
    if not save_intermediates:
        filtered_image = imf.get_filtered(filters)
    else:
        filtered_image, intermediates = imf.get_filtered(filters, return_intermediates=True)
        if not os.path.exists(os.path.splitext(out_filepath)[0]):
            os.mkdir(os.path.splitext(out_filepath)[0])
        for idx, intermediate in enumerate(intermediates):
            write_tif_slice(((intermediate.astype(np.float64) - intermediate.min()) / (intermediate.max() - intermediate.min()) * 255).astype(np.uint8), os.path.splitext(out_filepath)[0], '{:03d}.tif'.format(idx))

    write_tif_slice(filtered_image, *os.path.split(out_filepath))
