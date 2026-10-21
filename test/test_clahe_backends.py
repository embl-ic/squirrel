import unittest
import warnings

import numpy as np

from squirrel.library.filters import ImageFilter
from squirrel.library.normalization import clahe_on_image


class TestClaheBackends(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)
        rng = np.random.default_rng(12)
        self.image = rng.integers(1000, 55000, (96, 112), dtype=np.uint16)

    def test_skimage_bins_and_dtype(self):
        print('Testing CLAHE: skimage histogram resolution and dtype')
        outputs = [clahe_on_image(self.image, backend='skimage', histogram_bins=bins,
                                  clip_limit=0.01, tile_grid_size=(24, 24),
                                  tile_grid_in_pixels=True) for bins in (256, 1024)]
        for result in outputs:
            self.assertEqual(result.dtype, self.image.dtype)
            self.assertEqual(result.shape, self.image.shape)
        self.assertTrue(np.any(outputs[0] != outputs[1]))

    def test_opencv_default_unchanged(self):
        print('Testing CLAHE: default OpenCV backend remains unchanged')
        original = clahe_on_image(self.image, tile_grid_size=(4, 4))
        explicit = clahe_on_image(self.image, tile_grid_size=(4, 4), backend='opencv')
        np.testing.assert_array_equal(original, explicit)

    def test_pipeline_and_zero_handling(self):
        print('Testing CLAHE: pipeline skimage backend and zero preservation')
        image = self.image.copy()
        image[:8] = 0
        result = ImageFilter(image).get_filtered([
            ['clahe', {'backend': 'skimage', 'histogram_bins': 256,
                       'clip_limit': 0.01, 'tile_grid_size': [24, 24],
                       'tile_grid_in_pixels': True, 'keep_zeros': True,
                       'zeros_to_mean': True}]
        ])
        self.assertEqual(result.dtype, image.dtype)
        self.assertTrue(np.all(result[:8] == 0))
        self.assertTrue(np.any(result[8:] != image[8:]))

    def test_validation(self):
        print('Testing CLAHE: invalid backend and histogram options')
        invalid = [
            {'backend': 'other'},
            {'histogram_bins': 256},
            {'backend': 'skimage', 'histogram_bins': 1, 'clip_limit': 0.01},
            {'backend': 'skimage', 'histogram_bins': True, 'clip_limit': 0.01},
            {'backend': 'skimage', 'clip_limit': 3},
        ]
        for options in invalid:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    ImageFilter.validate_pipeline([['clahe', options]])
