import unittest
import warnings

import numpy as np


class TestFilterZerosToMeanDtype(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_same_preserves_dtype_and_replaces_zeros(self):
        print('Testing zeros_to_mean: preserve dtype and replace zeros')
        from squirrel.library.filters import ImageFilter

        for dtype in (np.uint8, np.uint16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                image = np.array([[0, 10], [20, 0]], dtype=dtype)
                result = ImageFilter.same(image, zeros_to_mean=True)
                self.assertEqual(result.dtype, image.dtype)
                np.testing.assert_array_equal(result, np.array([[15, 10], [20, 15]], dtype=dtype))

    def test_integer_mean_truncates_without_dtype_promotion(self):
        print('Testing zeros_to_mean: integer mean truncation')
        from squirrel.library.filters import ImageFilter

        for dtype in (np.uint8, np.uint16):
            with self.subTest(dtype=dtype):
                image = np.array([[0, 10, 11]], dtype=dtype)
                result = ImageFilter.same(image, zeros_to_mean=True)
                np.testing.assert_array_equal(result, np.array([[10, 10, 11]], dtype=dtype))
                self.assertEqual(result.dtype, image.dtype)

    def test_all_zeros_and_keep_zeros(self):
        print('Testing zeros_to_mean: all-zero images and keep_zeros')
        from squirrel.library.filters import ImageFilter

        image = np.zeros((3, 3), dtype=np.uint8)
        result = ImageFilter.same(image, zeros_to_mean=True, keep_zeros=True)
        np.testing.assert_array_equal(result, image)
        self.assertEqual(result.dtype, image.dtype)

        image = np.array([[0, 10], [20, 0]], dtype=np.uint8)
        result = ImageFilter.same(image, zeros_to_mean=True, keep_zeros=True)
        np.testing.assert_array_equal(result, image)

    def test_clahe_pipeline_accepts_uint8_and_uint16(self):
        print('Testing CLAHE pipeline: zeros_to_mean retains supported dtype')
        from squirrel.library.filters import ImageFilter

        for dtype in (np.uint8, np.uint16):
            with self.subTest(dtype=dtype):
                image = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
                image = (image % 200).astype(dtype)
                image[:4, :] = 0
                steps = [
                    ['same', {'normalize': True, 'normalization_method': 'quantile',
                              'quantiles': [0.05, 0.95], 'cast_dtype': dtype.__name__}],
                    ['clahe', {'zeros_to_mean': True, 'keep_zeros': True,
                               'tile_grid_size': [8, 8]}],
                ]
                result = ImageFilter(image).get_filtered(steps)
                self.assertEqual(result.dtype, np.dtype(dtype))
                self.assertTrue(np.all(result[:4, :] == 0))
