# Image filtering in squirrel

This guide documents the `ImageFilter` API and the filtering workflows. Filter names and parameter names below correspond to `squirrel/library/filters.py`.

## Quick start

```python
import numpy as np
from squirrel.library.filters import ImageFilter

image = np.zeros((256, 256), dtype=np.uint16)  # replace with your image
pipeline = [
    ['gaussian', {'sigma': 1.5}],
    ['median', {'radius': 2}],
]
filtered = ImageFilter(image).get_filtered(pipeline)
```

A pipeline is a list or tuple of two-element steps: `[filter_name, parameters_dict]`. Steps execute in order, and an empty pipeline returns the input image without filtering. Parameter dictionaries are also JSON-compatible when values use standard JSON types.

To inspect the installed filter registry:

```python
print(ImageFilter.get_available_filter_names())
ImageFilter.validate_pipeline(pipeline)  # validates without loading image data
```

## Available filters

| Name | Purpose | Filter-specific parameters (defaults) |
| --- | --- | --- |
| `gaussian` | Gaussian smoothing (VIGRA) | `sigma=1.0` |
| `gaussian_gradient_magnitude` | Gradient magnitude after Gaussian smoothing (VIGRA) | `sigma=1.0` |
| `median` | Median filtering (SciPy/scikit-image) | Exactly one of `radius` or `footprint`; `elliptical_footprint=False`, `on_binned=1`, `sample_footprint=None`, `random_seed=None` |
| `clahe` | Contrast-limited adaptive histogram equalization | `clip_limit=3.0`, `tile_grid_size=(127, 127)`, `invert_output=False`, `auto_mask=False`, `background_to_mean=False`, `tile_grid_in_pixels=False` |
| `vsnr` | Variational stationary noise removal | `filters=None`, `is_gpu=True`, `maxit=20`, `algo='auto'` |
| `fft_highpass` | FFT-based Gaussian high-pass | `sigma=(10, 10)` |
| `invert` | Invert intensity | `invert_range=None` |
| `same` | Identity/no-op filter | No filter-specific parameters |

Notes:

- `median` requires a 2D image. `radius` can be a nonnegative integer or a `(y, x)` pair; alternatively, provide a nonempty 2D `footprint`. `on_binned` filters a downsampled image and resizes the result; `sample_footprint` samples footprint locations, with `random_seed` enabling reproducible selection.
- `fft_highpass` requires a 2D image; its `sigma` is in `(y, x)` order and must be positive. The FFT assumes periodic image boundaries.
- `invert` without `invert_range` supports unsigned integer inputs; supply a two-value range for signed integer or floating-point images.
- `clahe` forwards its options to `squirrel.library.normalization.clahe_on_image`. `tile_grid_in_pixels` controls the interpretation of `tile_grid_size`; verify the intended tile size for your images.
- `vsnr` delegates to `squirrel.library.vsnr.vsnr_on_image`; GPU support depends on your installation.
- The underlying libraries and filters may impose additional image-dependent constraints. For example, VIGRA is used for Gaussian filters.

## Common processing options

Every filter accepts the following options in its parameter dictionary, in addition to filter-specific parameters:

| Option | Default | Meaning |
| --- | --- | --- |
| `filter_mode` | `'apply'` | `'apply'` uses the filter output; `'add'` adds it to the original image; `'subtract'` subtracts it from the original image. |
| `clip` | `None` | Optional `[minimum, maximum]` range applied after the filter/mode. |
| `normalize` | `False` | Scale the result using its finite minimum/maximum, or `normalization_range` when supplied. |
| `normalization_range` | `None` | Optional `[minimum, maximum]` range; requires `normalize=True`. |
| `cast_dtype` | `None` | Explicit destination NumPy dtype, e.g. `'uint16'` or `'float32'`. |
| `keep_zeros` | `False` | Restore originally zero-valued pixels to zero in the final output. |
| `zeros_to_mean` | `False` | Replace zero-valued input pixels with the mean of nonzero pixels *before* filtering. |

Normalization without `cast_dtype` produces `float32` values scaled to `[0, 1]`. With an integer `cast_dtype`, normalized values span that dtype's representable range. Without normalization or explicit casting, the filter's native result dtype is retained; it may differ from the input dtype. Integer casting clips out-of-range values rather than wrapping. Constant-valued data normalized without an explicit range produces zeros.

Example: normalize and cast a high-pass result to unsigned 16-bit:

```python
pipeline = [
    ['fft_highpass', {
        'sigma': [8, 8],
        'normalize': True,
        'cast_dtype': 'uint16',
        'keep_zeros': True,
    }],
]
```

`keep_zeros` and `zeros_to_mean` have different purposes and may be combined.

## Images, intermediate results, and stacks

For a single image:

```python
filtered, intermediate_images = ImageFilter(image).get_filtered(
    pipeline, return_intermediates=True, verbose=True
)
# intermediate_images contains one array per pipeline step.
```

For a stack, the first axis identifies independent slices:

```python
stack = np.zeros((16, 256, 256), dtype=np.uint16)  # (z, y, x)
result = ImageFilter(stack).get_filtered_stack(
    [['gaussian', {'sigma': 1.5}]],
    n_workers=4,
)
```

`get_filtered_stack` accepts nonempty 3D or 4D stacks, processes each item along axis 0 independently, and preserves slice order. Individual filters may impose stricter per-slice dimensionality constraints.

### Execution backends

`get_filtered_stack(..., n_workers=1, backend='threading')` supports:

| Backend | Behavior |
| --- | --- |
| `'threading'` | **Default.** Parallel threads when `n_workers > 1`; avoids transferring arrays between processes. |
| `'multiprocessing'` | Separate worker processes. Pipelines containing CLAHE use `spawn` workers with OpenCV native threading limited inside workers. |
| `'serial'` | Sequential execution; requires `n_workers=1`. |

All backends run sequentially when `n_workers=1`. `n_workers` must be a positive integer (not a boolean). Threads were faster than processes for Gaussian, median, CLAHE, and FFT high-pass in the project's benchmark workload, but actual performance depends on hardware, data, and filters.

```python
threaded = ImageFilter(stack).get_filtered_stack(pipeline, n_workers=4)
processes = ImageFilter(stack).get_filtered_stack(
    pipeline, n_workers=4, backend='multiprocessing'
)
serial = ImageFilter(stack).get_filtered_stack(
    pipeline, n_workers=1, backend='serial'
)
```

For multiprocessing on platforms using `spawn`, run application code under a `if __name__ == '__main__':` guard.

## File-based workflows

The 2D TIFF workflow is in `squirrel.workflows.image`:

```python
from squirrel.workflows.image import filter_2d_workflow

filter_2d_workflow(
    'input.tif',
    'filtered.tif',
    filters=[['gaussian', {'sigma': 1.5}]],
    save_intermediates=False,
)
```

Set `save_intermediates=True` to write normalized 8-bit previews of intermediate images into a directory based on the output filename.

The stack workflow is in `squirrel.workflows.volume`:

```python
from squirrel.workflows.volume import filter_2d_workflow as filter_stack_workflow

filter_stack_workflow(
    input_path='input_stack',
    out_path='filtered_stack',
    filters=[['median', {'radius': 2}]],
    n_workers=4,
    batch_size=16,
)
```

The stack workflow also accepts `key`, `pattern`, `out_key`, and `verbose`. `batch_size=None` processes the entire stack; batched output requires a directory or OME-Zarr destination. The workflow uses the default threading backend internally; to select multiprocessing explicitly, call `ImageFilter.get_filtered_stack` directly.

## Errors and practical guidance

Pipelines are validated before filtering begins. Invalid parameter types generally raise `TypeError`; invalid values or unsupported pipeline steps raise `ValueError`. Pipeline execution errors identify the failing step and filter. Input images must be nonempty arrays of real numeric dtype; complex and object arrays are not supported.

A few practical examples:

```python
# Suppress slow-varying background
highpass = [['fft_highpass', {'sigma': [10, 10]}]]

# Smooth and enhance local contrast
contrast = [
    ['gaussian', {'sigma': 1.0}],
    ['clahe', {'clip_limit': 3.0, 'tile_grid_size': [8, 8]}],
]

# Median filtering with reproducible footprint sampling
sampled_median = [['median', {
    'radius': 3, 'sample_footprint': 15, 'random_seed': 42
}]]
```

For repeatable scientific workflows, record the pipeline parameters, source dtype, output dtype, and the squirrel version used.
