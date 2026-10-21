import unittest
import warnings

import numpy as np


class TestPeakNormalization(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_peak_alignment_across_slices(self):
        print('Testing peak normalization: align mode and FWHM')
        from squirrel.library.normalization import get_histogram_peak, normalize_image

        rng = np.random.default_rng(17)
        reference = rng.normal(size=(512, 512))
        outputs = []
        for center, width in ((80, 12), (140, 25)):
            image = reference * width + center
            outputs.append(normalize_image(image, method='peak', peak_target=0.5,
                                           peak_width_target=0.2))
        peaks = [get_histogram_peak(result) for result in outputs]
        for center, width in peaks:
            self.assertAlmostEqual(center, 0.5, delta=0.015)
            self.assertAlmostEqual(width, 0.2, delta=0.025)

    def test_peak_ignores_separate_charging_population(self):
        print('Testing peak normalization: separated charging population')
        from squirrel.library.normalization import get_histogram_peak

        rng = np.random.default_rng(5)
        tissue = rng.normal(100, 12, 180000)
        charging = rng.normal(190, 8, 20000)
        center, width = get_histogram_peak(np.concatenate((tissue, charging)))
        self.assertAlmostEqual(center, 100, delta=3)
        self.assertAlmostEqual(width, 28.3, delta=5)

    def test_peak_degenerate_and_invalid_parameters(self):
        print('Testing peak normalization: degenerate data and validation')
        from squirrel.library.normalization import normalize_image

        image = np.ones((16, 16))
        np.testing.assert_array_equal(normalize_image(image, method='peak'), 0)
        with self.assertRaisesRegex(ValueError, 'peak_width_target'):
            normalize_image(image, method='peak', peak_width_target=0)
        with self.assertRaisesRegex(ValueError, 'peak_search_range'):
            normalize_image(image, method='peak', peak_search_range=(10, 5))

    def test_peak_filter_wrapper(self):
        print('Testing peak normalization through filter wrapper')
        from squirrel.library.filters import ImageFilter

        rng = np.random.default_rng(7)
        image = rng.normal(100, 15, (128, 128)).astype('float32')
        pipeline = [['gaussian', {'sigma': 0.5, 'normalize': True,
                                  'normalization_method': 'peak',
                                  'peak_target': 0.5, 'peak_width_target': 0.2}]]
        ImageFilter.validate_pipeline(pipeline)
