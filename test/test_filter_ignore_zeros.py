import unittest
import warnings

import numpy as np


class TestFilterIgnoreZeros(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_quantile_all_flag_combinations(self):
        print('Testing quantile normalization: independent zero options')
        from squirrel.library.filters import ImageFilter

        image = np.array([[0, 0, 0, 10, 20, 30]], dtype=np.uint8)
        for keep_zeros in (False, True):
            for ignore_zeros in (False, True):
                with self.subTest(keep_zeros=keep_zeros, ignore_zeros=ignore_zeros):
                    actual = ImageFilter.same(
                        image, normalize=True, normalization_method='quantile',
                        quantiles=(0.25, 0.75), cast_dtype='uint8',
                        keep_zeros=keep_zeros, ignore_zeros=ignore_zeros)
                    from squirrel.library.normalization import normalize_image
                    expected = normalize_image(
                        image, method='quantile', quantiles=(0.25, 0.75),
                        output_dtype='uint8', ignore_zeros=ignore_zeros,
                        keep_zeros=keep_zeros)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(actual.dtype, np.dtype('uint8'))
                    if keep_zeros:
                        self.assertTrue(np.all(actual[image == 0] == 0))
                    self.assertEqual(actual[0, 3], 0 if ignore_zeros else 145)

    def test_minmax_and_new_zeros(self):
        print('Testing minmax normalization: ignore zeros in filtered result')
        from squirrel.library.filters import ImageFilter

        image = np.array([[0, 10, 20, 30]], dtype=np.uint8)
        result = ImageFilter.same(image, normalize=True, ignore_zeros=True,
                                  keep_zeros=True, cast_dtype='uint8')
        np.testing.assert_array_equal(result, [[0, 0, 127, 255]])

    def test_pipeline_accepts_ignore_zeros(self):
        print('Testing pipeline: ignore_zeros validation and execution')
        from squirrel.library.filters import ImageFilter

        image = np.array([[0, 10, 20, 30]], dtype=np.uint8)
        steps = [['same', {'normalize': True, 'ignore_zeros': True,
                           'keep_zeros': True, 'cast_dtype': 'uint8'}]]
        self.assertEqual(len(ImageFilter.validate_pipeline(steps)), 1)
        np.testing.assert_array_equal(ImageFilter(image).get_filtered(steps), [[0, 0, 127, 255]])

    def test_invalid_ignore_zeros_type(self):
        print('Testing ignore_zeros: reject nonboolean options')
        from squirrel.library.filters import ImageFilter

        image = np.array([[0, 10]], dtype=np.uint8)
        with self.assertRaises((TypeError, ValueError)):
            ImageFilter.same(image, normalize=True, ignore_zeros='yes')
        with self.assertRaises((TypeError, ValueError)):
            ImageFilter.validate_pipeline([['same', {'ignore_zeros': 1}]])

    def test_explicit_range_is_unaffected(self):
        print('Testing ignore_zeros: explicit normalization range')
        from squirrel.library.filters import ImageFilter

        image = np.array([[0, 10, 20]], dtype=np.uint8)
        results = [ImageFilter.same(image, normalize=True,
                                    normalization_range=(0, 20), cast_dtype='uint8',
                                    ignore_zeros=value) for value in (False, True)]
        np.testing.assert_array_equal(results[0], results[1])
