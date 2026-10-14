
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
    ImageFilter.validate_pipeline(filters)

    from squirrel.library.io import read_tif_slice, write_tif_slice
    img = read_tif_slice(input_filepath, return_filepath=False)

    imf = ImageFilter(img)
    if not save_intermediates:
        filtered_image = imf.get_filtered(filters, verbose=verbose)
    else:
        filtered_image, intermediates = imf.get_filtered(filters, return_intermediates=True, verbose=verbose)
        intermediate_dir = os.path.splitext(out_filepath)[0]
        # Prepare every preview before creating the directory or writing files.
        previews = []
        for intermediate in intermediates:
            values = intermediate.astype(np.float64)
            low, high = values.min(), values.max()
            if high == low:
                preview = np.zeros(values.shape, dtype=np.uint8)
            else:
                preview = ((values - low) / (high - low) * 255).astype(np.uint8)
            previews.append(preview)
        os.makedirs(intermediate_dir, exist_ok=True)
        for idx, preview in enumerate(previews):
            write_tif_slice(preview, intermediate_dir, '{:03d}.tif'.format(idx))

    write_tif_slice(filtered_image, *os.path.split(out_filepath))
