import unittest
import warnings

import numpy as np

from squirrel.library.filters import ImageFilter
from squirrel.library.normalization import get_normalization_limits, normalize_image


class TestNormalizationCore(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_get_normalization_limits_quantiles(self):
        print('Testing get_normalization_limits: quantile estimation')
        image = np.arange(101, dtype=np.uint8).reshape(1, -1)
        self.assertEqual(get_normalization_limits(image, 'quantile', (0.1, 0.9)), (10., 90.))

    def test_normalize_image_anchors(self):
        print('Testing normalize_image: quantile anchors')
        image = np.arange(101, dtype=np.uint8).reshape(1, -1)
        result = normalize_image(image, 'quantile', (0.1, 0.9), anchors=(0.2, 0.8))
        np.testing.assert_allclose(result[0, [10, 90]], [0.2, 0.8], atol=1e-6)
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 1)

    def test_normalize_image_mask_and_zeros(self):
        print('Testing normalize_image: mask and zero preservation')
        image = np.array([[0, 10, 20, 100]], dtype=np.uint16)
        mask = np.array([[False, True, True, False]])
        self.assertEqual(get_normalization_limits(image, mask=mask), (10., 20.))
        result = normalize_image(image, ignore_zeros=True, keep_zeros=True, output_dtype='uint8')
        np.testing.assert_array_equal(result, [[0, 0, 28, 255]])

    def test_normalization_integer_masks(self):
        print('Testing normalization: integer masks select nonzero pixels')
        image = np.array([[0, 10, 20, 100]], dtype=np.uint16)
        expected_limits = (10., 20.)
        for dtype, foreground in [('uint8', 1), ('uint8', 255), ('int16', -1)]:
            with self.subTest(dtype=dtype, foreground=foreground):
                mask = np.array([[0, foreground, foreground, 0]], dtype=dtype)
                self.assertEqual(get_normalization_limits(image, mask=mask), expected_limits)
                result = normalize_image(image, mask=mask, output_dtype='uint8')
                np.testing.assert_array_equal(result, [[0, 0, 255, 255]])

    def test_normalization_invalid_masks(self):
        print('Testing normalization: invalid mask shape and dtype')
        image = np.array([[0, 10, 20]], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, 'mask must match image shape'):
            get_normalization_limits(image, mask=np.ones(3, dtype=np.uint8))
        with self.assertRaisesRegex(TypeError, 'mask must be boolean or integer'):
            get_normalization_limits(image, mask=np.ones(image.shape, dtype=np.float32))

    def test_normalize_image_degenerate(self):
        print('Testing normalize_image: constant and empty images')
        for image in (np.zeros((3, 3)), np.ones((3, 3))):
            with self.subTest(value=image[0, 0]):
                self.assertFalse(normalize_image(image, method='quantile').any())

    def test_filter_pipeline_quantiles(self):
        print('Testing ImageFilter: quantile normalization pipeline')
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        pipeline = [['same', {'normalize': True, 'normalization_method': 'quantile',
                              'quantiles': [0.1, 0.9], 'cast_dtype': 'uint8'}]]
        ImageFilter.validate_pipeline(pipeline)
        result = ImageFilter(image).get_filtered(pipeline)
        self.assertEqual(result.dtype, np.dtype('uint8'))
        self.assertEqual((result.min(), result.max()), (0, 255))

    def test_filter_pipeline_invalid_options(self):
        print('Testing ImageFilter: invalid normalization options')
        cases = [
            {'normalization_method': 'other', 'normalize': True},
            {'quantiles': [0.9, 0.1], 'normalize': True},
            {'anchors': [-0.1, 0.9], 'normalize': True},
            {'normalization_method': 'quantile'},
        ]
        for parameters in cases:
            with self.subTest(parameters=parameters):
                with self.assertRaises((ValueError, TypeError)):
                    ImageFilter.validate_pipeline([['same', parameters]])
