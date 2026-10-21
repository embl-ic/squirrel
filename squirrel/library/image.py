
import numpy as np


def get_bounds(image, return_ints=False):
    # coordinates of nonzero points
    true_points = np.argwhere(image)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)

    if return_ints:
        # return [min0, min1, ..., max0+1, max1+1, ...]
        return np.concatenate([top_left, bottom_right + 1]).astype(float).tolist()
    else:
        # build a tuple of slice objects, one per dimension
        bounds = tuple(slice(t, b + 1) for t, b in zip(top_left, bottom_right))
        return bounds


def get_bounds_of_stack(stack_h, stack_shape, return_ints=False, z_range=None):

    if z_range is None:
        z_range = [0, stack_shape[0]]

    return [
        get_bounds(stack_h[idx], return_ints)
        for idx in range(*z_range)
    ]


# def apply_auto_pad(transforms, stack_shape, stack_bounds, extra_padding=0):

#     def _transform_on_bounds(t, b):

#         t_ = t.astype('float64')

#         min_yx = [b[0], b[1]]
#         min_y_max_x = [b[0], b[3]]
#         max_y_min_x = [b[2], b[1]]
#         max_yx = [b[2], b[3]]

#         t_min_yx = np.matmul(np.linalg.inv(t_), min_yx + [1.])
#         t_min_y_max_x = np.matmul(np.linalg.inv(t_), min_y_max_x + [1.])
#         t_max_y_min_x = np.matmul(np.linalg.inv(t_), max_y_min_x + [1.])
#         t_max_yx = np.matmul(np.linalg.inv(t_), max_yx + [1.])

#         new_b = np.array([
#             np.min([t_min_yx, t_min_y_max_x, t_max_y_min_x, t_max_yx], axis=0),
#             np.max([t_min_yx, t_min_y_max_x, t_max_y_min_x, t_max_yx], axis=0)
#         ])[:, :2]

#         return new_b

#     new_bounds = np.array([
#         _transform_on_bounds(matrix.get_matrix('Ms'), stack_bounds[idx])
#         for idx, matrix in enumerate(transforms)
#     ])

#     new_bounds = [
#         np.min(new_bounds[:, 0], axis=0),
#         np.max(new_bounds[:, 1], axis=0)
#     ]

#     # Modify the offsets within the transforms to move everything towards the origin
#     from ..library.affine_matrices import AffineMatrix
#     from ..library.transformation import setup_translation_matrix
#     new_transforms = []
#     for matrix in transforms:
#         new_transforms.append(
#             matrix * AffineMatrix(parameters=setup_translation_matrix(new_bounds[0] - extra_padding, ndim=2).flatten())
#         )
#     transforms.update_stack(new_transforms)

#     # Also modify the stack_shape now to crop or extend the images
#     stack_shape[1:] = (new_bounds[1] - new_bounds[0] + 2 * extra_padding).astype(int).tolist()

#     return transforms, stack_shape


def assert_equal_shape(im1, im2):
    # Compute target shape as elementwise max
    new_shape = tuple(max(s1, s2) for s1, s2 in zip(im1.shape, im2.shape))

    # Pad each image as needed
    def pad_to_shape(im, target_shape):
        pad_width = [(0, t - s) for s, t in zip(im.shape, target_shape)]
        return np.pad(im, pad_width, mode='constant', constant_values=0)

    im1_padded = pad_to_shape(im1, new_shape)
    im2_padded = pad_to_shape(im2, new_shape)

    return im1_padded, im2_padded


def image_to_shape(image, shape):

    image_shape = np.array(image.shape)
    shape = np.ceil(np.array(shape)).astype(int)

    max_shape = (
        max(image_shape[0], shape[0]),
        max(image_shape[1], shape[1])
    )

    new_image = np.zeros(max_shape, dtype=image.dtype)
    try:
        s = np.s_[
            :max_shape[0],
            :max_shape[1]
        ]
        new_image[s] = image[s]
        return new_image
    except ValueError:
        pass
    try:
        s = np.s_[
            :image_shape[0],
            :max_shape[1]
        ]
        new_image[s] = image[s]
        return new_image
    except ValueError:
        pass
    try:
        s = np.s_[
            :max_shape[0],
            :image_shape[1]
        ]
        new_image[s] = image[s]
        return new_image
    except ValueError:
        pass
    try:
        s = np.s_[
            :image_shape[0],
            :image_shape[1]
        ]
        new_image[s] = image[s]
        return new_image
    except ValueError:
        pass


def _get_default_font(font_size):
    from PIL import ImageFont
    import matplotlib.font_manager as fm

    try:
        font_path = fm.findSystemFonts(fontpaths=None, fontext='ttf')[0]  # Get first available font
        return ImageFont.truetype(font_path, font_size)
    except IndexError:
        return ImageFont.load_default()  # Fallback


def _get_scaled_font(font_size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)  # Scalable font
    except IOError:
        return _get_default_font(font_size)


def draw_strings_on_image(img_filepath, out_filepath, strings, positions, font_size=30, color=(255, 0, 0), pivot='center', verbose=False):

    assert len(strings) == len(positions), 'Number of strings and positions must match!'

    from PIL import Image, ImageDraw, ImageFont
    # Load the image
    img = Image.open(img_filepath).convert("RGBA")

    # Create a drawing context
    draw = ImageDraw.Draw(img)

    if pivot == 'center':
        positions += (np.array(img.size) / 2)
    if verbose:
        print(f'positions = {positions}')

    font = _get_scaled_font(font_size)

    # Draw each number at its given position
    for idx, string in enumerate(strings):

        # Get text bounding box to determine size
        bbox = font.getbbox(string)  # (left, top, right, bottom)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Compute the new top-left position for centering
        pos = positions[idx]
        pos_ = [pos[0] - text_width / 2, pos[1] - text_height / 4 * 3]

        draw.text(pos_, string, font=font, fill=color)
        draw.rectangle([tuple((pos - font_size/1.6).tolist()), tuple((pos + font_size/1.6).tolist())], outline=(0, 255, 255), width=2)

    # Save the modified image
    img.save(out_filepath, "PNG")


def fft_highpass(image: np.ndarray, sigma=(10, 10)) -> np.ndarray:
    """FFT Gaussian high-pass on a 2D periodic image; sigma is (y, x)."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError('fft_highpass expects a 2D image')
    if np.isscalar(sigma):
        sigma = (sigma, sigma)
    if len(sigma) != 2 or not np.all(np.isfinite(sigma)) or np.any(np.asarray(sigma) <= 0):
        raise ValueError('sigma must contain two positive finite values')
    sigma_y, sigma_x = sigma
    ky = np.fft.fftfreq(image.shape[0])[:, None]
    kx = np.fft.fftfreq(image.shape[1])[None, :]
    response = -np.expm1(-2 * np.pi**2 * (sigma_y**2 * ky**2 + sigma_x**2 * kx**2))
    return np.fft.ifft2(np.fft.fft2(image) * response).real


def median_filter_2d(in_array, radius=None, footprint=None, elliptical_footprint=False,
                     on_binned=1, sample_footprint=None, random_seed=None):
    from scipy.ndimage import median_filter, zoom
    if (radius is None) == (footprint is None):
        raise ValueError('Supply exactly one of radius and footprint')
    if not isinstance(on_binned, (int, np.integer)) or on_binned < 1:
        raise ValueError('on_binned must be a positive integer')
    if in_array.ndim != 2:
        raise ValueError('median currently expects a 2D image')
    original_shape = in_array.shape
    if on_binned != 1:
        from squirrel.library.scaling import average_bin_image
        in_array = average_bin_image(in_array, on_binned)
    if radius is not None:
        if np.isscalar(radius):
            radius = (radius, radius)
        if len(radius) != 2 or any(not isinstance(r, (int, np.integer)) or r < 0 for r in radius):
            raise ValueError('radius must be nonnegative integer or pair of nonnegative integers')
        ry, rx = radius
        if elliptical_footprint:
            y, x = np.ogrid[-ry:ry + 1, -rx:rx + 1]
            footprint = ((y / max(ry, 1))**2 + (x / max(rx, 1))**2 <= 1)
        elif sample_footprint is not None:
            footprint = np.ones((2 * ry + 1, 2 * rx + 1), dtype=bool)
        else:
            filtered = median_filter(in_array, size=(2 * ry + 1, 2 * rx + 1))
            footprint = None
    if footprint is not None:
        footprint = np.asarray(footprint, dtype=bool)
        if footprint.ndim != 2 or not np.any(footprint):
            raise ValueError('footprint must be a nonempty 2D boolean mask')
        if sample_footprint is not None:
            if not isinstance(sample_footprint, (int, np.integer)) or sample_footprint < 1:
                raise ValueError('sample_footprint must be a positive integer')
            coords = np.argwhere(footprint)
            if sample_footprint < len(coords):
                rng = np.random.default_rng(random_seed)
                chosen = coords[rng.choice(len(coords), size=sample_footprint, replace=False)]
                footprint = np.zeros_like(footprint)
                footprint[tuple(chosen.T)] = True
        from skimage.filters import median as sk_median
        filtered = sk_median(in_array, footprint=footprint)
    if on_binned != 1:
        # Explicit target shape avoids unfilled zero borders for nondivisible dimensions.
        factors = [target / source for target, source in zip(original_shape, filtered.shape)]
        filtered = zoom(filtered, factors, order=3, mode='nearest', grid_mode=True)
        filtered = filtered[tuple(slice(0, n) for n in original_shape)]
    return filtered
