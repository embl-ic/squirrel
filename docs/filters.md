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
| `normalize` | `False` | Normalize the filtered result after filtering, `filter_mode`, and `clip`. |
| `normalization_method` | `'minmax'` | Choose `'minmax'`, `'quantile'`, or `'peak'` to estimate input intensity limits. |
| `peak_target` | `0.5` | Target dominant histogram peak intensity, as a fraction of output range. |
| `peak_width_target` | `0.15` | Target peak FWHM, in fractional output intensity units. |
| `peak_smoothing_sigma` | `3.0` | Gaussian smoothing sigma in histogram bins. |
| `peak_search_range` | `None` | Optional input-intensity interval for searching the dominant peak. |
| `quantiles` | `(0.01, 0.99)` | Lower and upper quantile fractions for `'quantile'` normalization. |
| `anchors` | `(0.0, 1.0)` | Fractions of the output intensity range to which the input limits are mapped. |
| `normalization_range` | `None` | Optional `[minimum, maximum]` range; requires `normalize=True`. |
| `cast_dtype` | `None` | Explicit destination NumPy dtype, e.g. `'uint16'` or `'float32'`. |
| `rescale_dtype` | `None` | Rescale the full input dtype range to `uint8` or `uint16` without per-image statistics; requires `uint8` or `uint16` input and cannot be combined with `normalize=True` or `cast_dtype`. |
| `keep_zeros` | `False` | Restore originally zero-valued pixels to zero after processing; does not affect normalization statistics. |
| `ignore_zeros` | `False` | Exclude zero-valued pixels in the filtered result from automatic normalization limit estimation; does not preserve zeros in the output. |
| `zeros_to_mean` | `False` | Replace zero-valued input pixels with the mean of nonzero pixels *before* filtering. |

Normalization without `cast_dtype` produces `float32` values scaled to `[0, 1]`. With an integer `cast_dtype`, normalized values span that dtype's representable range. Without normalization or explicit casting, the filter's native result dtype is retained; it may differ from the input dtype. Integer casting clips out-of-range values rather than wrapping. Constant-valued data normalized without an explicit range produces zeros. `rescale_dtype` is independent of normalization and cannot be combined with `normalize=True` or `cast_dtype`.

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

`keep_zeros`, `ignore_zeros`, and `zeros_to_mean` have different purposes and may be combined. `keep_zeros` restores the input image's zero pixels after filtering; `ignore_zeros` excludes zero-valued pixels in the result being normalized from automatic limit estimation. Neither option implies the other. With `keep_zeros=True, ignore_zeros=False`, zeros **are included** in the statistics (unless an explicit normalization range is supplied). With `keep_zeros=True, ignore_zeros=True`, zero-valued pixels are excluded from estimation and original zeros are restored. `ignore_zeros` does not affect a user-supplied `normalization_range`.

For background-masked images, an explicit combination is recommended:

```python
pipeline = [
    ['same', {
        'normalize': True,
        'normalization_method': 'quantile',
        'quantiles': [0.05, 0.95],
        'anchors': [0.1, 0.9],
        'cast_dtype': 'uint8',
        'keep_zeros': True,
        'ignore_zeros': True,
    }],
]
```

This is a behavior change for pipelines previously using `keep_zeros=True` alone: zero pixels now contribute to automatic normalization statistics unless `ignore_zeros=True` is also set.

`normalization_range` overrides automatic limit estimation, irrespective of `normalization_method`. Quantiles and anchors must each be strictly increasing pairs in `[0, 1]`. Nondefault normalization options require `normalize=True`. Normalization is performed **per image**; `get_filtered_stack` therefore estimates separate limits for each slice. A constant image or an image with no eligible pixels produces zeros when limits are estimated automatically.

### Fixed-range dtype rescaling (`rescale_dtype`)

Use `rescale_dtype` to map the **full representable range** of a `uint8` or
`uint16` image to the full range of the requested unsigned integer dtype.
Unlike `normalize=True`, it does **not** use per-image minima, maxima, or
quantiles, so the same input value maps to the same output value in every
stack slice. It is useful as the last step of a 16-bit filtering pipeline:

```python
pipeline = [
    ['gaussian', {'sigma': 0.75, 'cast_dtype': 'uint16'}],
    ['same', {'rescale_dtype': 'uint8'}],
]
```

The equivalent CLI filter argument is:

```bash
'["same", {"rescale_dtype": "uint8"}]'
```

For `uint16` to `uint8`, the mapping is `round(value * 255 / 65535)`:
`0 → 0`, `65535 → 255`. The reverse `uint8` to `uint16` mapping uses
`round(value * 65535 / 255)`; converting to the same dtype preserves values.
Rounding is to the nearest integer. The mapping is **not** stretched to the
observed pixel range: a slice with maximum 30000 will not be mapped to 255.

Only `uint8` and `uint16` **input and output** dtypes are supported.
`rescale_dtype` cannot be combined with `normalize=True` or with `cast_dtype`;
invalid combinations raise errors describing the conflict and how to fix it.
In particular, do not use `cast_dtype='uint8'` as a substitute: it clips
16-bit values above 255 instead of rescaling them. If a preceding filter
produces floating-point output, first explicitly convert it to `uint16`
(e.g. with `cast_dtype='uint16'` on that preceding step) before rescaling.

### Quantile normalization examples

For direct filter calls, quantiles specify which input intensities are mapped to the output anchors:

```python
from squirrel.library.filters import ImageFilter

result = ImageFilter.same(
    image, normalize=True, normalization_method='quantile',
    quantiles=(0.01, 0.99), anchors=(0.0, 1.0), cast_dtype='uint16'
)
```

The same options work in JSON-compatible pipelines:

```python
pipeline = [
    ['gaussian', {
        'sigma': 1.5, 'normalize': True,
        'normalization_method': 'quantile',
        'quantiles': [0.05, 0.95],
        'anchors': [0.1, 0.9],
        'cast_dtype': 'uint8',
    }],
]
```

Here, the 5th and 95th percentiles are mapped to approximately 10% and 90% of the `uint8` range, respectively; values beyond those limits are clipped to the destination dtype range. Anchors `(0.0, 1.0)` map the chosen limits to the full range.

### Shared normalization helpers

`get_normalization_limits(image, method='minmax', quantiles=(0.01, 0.99), mask=None, ignore_zeros=False)` estimates a `(low, high)` pair from finite pixels, returning `None` for an empty or constant selection. `mask` is a boolean array matching the image shape and **only controls which pixels are used to estimate limits**. `ignore_zeros=True` also excludes zeros from estimation.

`normalize_image(image, method='minmax', quantiles=(0.01, 0.99), normalization_range=None, anchors=(0.0, 1.0), output_dtype='float32', mask=None, ignore_zeros=False, keep_zeros=False)` uses those limits for linear rescaling. Unlike `ignore_zeros`, `keep_zeros` also forces originally zero-valued pixels to zero in the result. For integer output, the full representable dtype range is used; floating output uses `[0, 1]`.

```python
from squirrel.library.normalization import normalize_image

normalized = normalize_image(
    image, method='quantile', quantiles=(0.01, 0.99),
    ignore_zeros=True, keep_zeros=True, output_dtype='uint8'
)
```

Legacy functions such as `norm_full_range`, `apply_quantiles`, and `normalize_slices` remain available. They retain their application-specific defaults, so their outputs need not match a bare `normalize_image()` call. CLAHE is a separate local contrast enhancement operation, not a quantile normalization method.


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
    backend='multiprocessing',
    batch_size=16,
)
```

The stack workflow also accepts `key`, `pattern`, `out_key`, and `verbose`. `batch_size=None` processes the entire stack; batched output requires a directory or OME-Zarr destination. Select `backend="threading"` (default), `backend="multiprocessing"`, or `backend="serial"` (requires `n_workers=1`). The CLI exposes `--backend` and `--n_workers`; for example, `sq-stack-filter_2d_workflow input output --backend multiprocessing --n_workers 4 -f '["same", {}]'`. Parallel execution is across slices, not individual filters. With one worker execution is serial regardless of backend. Multiprocessing incurs process and data-transfer overhead, so benchmark on representative stacks.

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

### Stack filtering progress

`ImageFilter.get_filtered_stack(..., progress=True)` displays a `tqdm` progress bar
with the number of completed slices, elapsed time, processing rate, and estimated
remaining time. This works for serial, threading, and multiprocessing backends.
Use `progress=False` to suppress the bar (for example, in automated scripts).
Progress is updated in the main process as results are consumed in slice order.

The file-based stack workflow `squirrel.workflows.volume.filter_2d_workflow`
also accepts `progress=True` by default. Batched runs display one continuous
slice-level bar across all batches, rather than a separate bar per batch.
`verbose=True` additionally prints stack configuration, batch operations, and
output-writing messages. The progress bar reports filtering, not writing time.

```python
filtered = ImageFilter(stack).get_filtered_stack(
    pipeline, n_workers=4, backend='threading', progress=True
)
```

### Volume intermediate previews

Pass `save_intermediates=True` to `filter_2d_workflow`, or
`--save_intermediates` to the `filter_2d` stack command, to write Fiji-compatible
8-bit previews after each filter step. For a TIFF stack written to
`output_stack/`, previews are saved to
`output_stack_intermediates/filter00_<name>/slice_00000.tif`, etc.
Each slice is min-max scaled independently to uint8, as in the image
workflow. Constant-valued images become all-zero previews. The final
stack retains its original filtered dtype. Preview generation works with
batched and unbatched processing and is disabled by default.

### Peak normalization (experimental)

`normalization_method='peak'` detects the dominant mode in a 256-bin histogram,
smooths it, measures its full width at half maximum (FWHM), and applies an
affine intensity transform mapping its center to `peak_target` and its FWHM
to `peak_width_target`. Output is clipped to the target dtype range. This
normalizes peak position and width, **not histogram peak height**.

```python
normalized = normalize_image(
    image, method='peak', peak_target=0.5, peak_width_target=0.15,
    peak_smoothing_sigma=3.0, peak_search_range=None
)
```

Peak normalization works per slice and does not yet regularize parameters
along Z. `anchors` must remain `(0.0, 1.0)` and `normalization_range`
is not supported for this method. Constant images or peaks with no
measurable half-maximum crossings return zeros. Strong charging may
still cause the wrong peak to be selected; `peak_search_range` can help.

### CLAHE backends and histogram bins

The default `backend="opencv"` retains existing CLAHE behavior. Set
`backend="skimage"` to use scikit-image CLAHE and specify `histogram_bins`
(default 256). The scikit-image `clip_limit` is normalized to [0, 1] and
**is not numerically equivalent** to OpenCV's `clip_limit`. For example:

```python
['clahe', {'backend': 'skimage', 'histogram_bins': 256,
           'clip_limit': 0.01, 'tile_grid_in_pixels': True,
           'tile_grid_size': [63, 63]}]
```

With `tile_grid_in_pixels=True`, the scikit-image backend uses the specified
size as its kernel dimensions. Otherwise, `tile_grid_size` is interpreted as
OpenCV-style tile counts and converted to approximate kernel dimensions.
Scikit-image produces normalized floating-point output, which squirrel scales
back to the original integer dtype. Its internal quantization may not preserve
all 16-bit input levels, even when `histogram_bins=65536`.
