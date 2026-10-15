import unittest
import warnings

import numpy as np


class TestNormalizationClaheDtype(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)
        self.image = np.arange(64, dtype=np.uint8).reshape(8, 8)

    def test_clahe_on_image_rejects_float_input(self):
        print('Testing clahe_on_image: informative error for floating-point input')
        from squirrel.library.normalization import clahe_on_image

        with self.assertRaisesRegex(ValueError, '2D uint8 or uint16.*Normalize and cast'):
            clahe_on_image(self.image.astype('float32'), tile_grid_size=(2, 2))

    def test_clahe_on_image_rejects_rgb_input(self):
        print('Testing clahe_on_image: informative error for multichannel input')
        from squirrel.library.normalization import clahe_on_image

        with self.assertRaisesRegex(ValueError, '2D uint8 or uint16'):
            clahe_on_image(np.repeat(self.image[..., None], 3, axis=2))

    def test_clahe_on_image_accepts_supported_dtypes(self):
        print('Testing clahe_on_image: uint8 and uint16 inputs')
        from squirrel.library.normalization import clahe_on_image

        for dtype in ('uint8', 'uint16'):
            with self.subTest(dtype=dtype):
                image = self.image.astype(dtype)
                result = clahe_on_image(image, tile_grid_size=(2, 2))
                self.assertEqual(result.dtype, image.dtype)
                self.assertEqual(result.shape, image.shape)

    def test_filter_pipeline_reports_clahe_dtype_error(self):
        print('Testing ImageFilter: CLAHE dtype error includes pipeline context')
        from squirrel.library.filters import ImageFilter

        image = self.image.astype('float32')
        with self.assertRaisesRegex(ValueError, 'Filter step 0.*CLAHE requires'):
            ImageFilter(image).get_filtered([['clahe', {'tile_grid_size': (2, 2)}]])
