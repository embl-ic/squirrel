import unittest
import warnings

import numpy as np

from squirrel.library.data import get_contrast_limits, norm_8bit, norm_full_range
from squirrel.library.normalization import apply_quantiles, _adjust_greyscale_in_image


class TestNormalizationLegacy(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_norm_8bit_quantiles(self):
        print('Testing norm_8bit: quantile normalization')
        image = np.array([0, 10, 20, 30, 40], dtype='uint8')
        np.testing.assert_array_equal(norm_8bit(image, (0.25, 0.75)), [0, 0, 127, 255, 255])

    def test_norm_full_range_mask_and_dtype(self):
        print('Testing norm_full_range: masked quantiles and output dtype')
        image = np.array([0, 10, 20, 30, 100], dtype='uint16')
        mask = np.array([False, True, True, True, False])
        result = norm_full_range(image, (0.0, 1.0), mask=mask, cast_8bit=True)
        np.testing.assert_array_equal(result, [0, 0, 127, 255, 0])
        self.assertEqual(result.dtype, np.dtype('uint8'))

    def test_norm_full_range_default_anchors(self):
        print('Testing norm_full_range: default quantile anchors')
        image = np.arange(101, dtype='uint8')
        result = norm_full_range(image, (0.1, 0.9))
        self.assertEqual(result[10], 25)
        self.assertEqual(result[90], 229)

    def test_apply_quantiles_background_and_zeros(self):
        print('Testing apply_quantiles: background exclusion and zero preservation')
        image = np.array([0, 2, 10, 20, 254, 255], dtype='uint8')
        result = apply_quantiles(image, quantiles=(0., 1.), anchors=(0.2, 0.8), keep_zeros=True)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[2], 119)
        self.assertEqual(result[3], 204)

    def test_adjust_greyscale_range(self):
        print('Testing adjust_greyscale: custom grey levels')
        image = np.array([0, 10, 20, 30], dtype='uint8')
        result = _adjust_greyscale_in_image(image, (10, 20), (50, 200))
        np.testing.assert_array_equal(result, [0, 50, 200, 255])

    def test_get_contrast_limits(self):
        print('Testing get_contrast_limits: minmax compatibility')
        image = np.array([2, 7, 9], dtype='uint8')
        self.assertEqual(get_contrast_limits(image), (2, 9))
        self.assertIsNone(get_contrast_limits(image, 'unsupported'))
