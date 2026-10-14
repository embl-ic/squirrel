import unittest
import warnings
import numpy as np


class TestImageFilter(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_cast_without_normalization(self):
        print('Testing dtype casting without normalization ...')
        from squirrel.library.filters import ImageFilter
        img = np.array([100., 200.])
        result = ImageFilter(img).get_filtered([['same', dict(cast_dtype='uint16')]])
        np.testing.assert_array_equal(result, [100, 200])

    def test_normalization_float_and_integer(self):
        print('Testing float and integer normalization ...')
        from squirrel.library.filters import ImageFilter
        img = np.array([100., 200.])
        imf = ImageFilter(img)
        result_float = imf.get_filtered([['same', dict(normalize=True)]])
        result_uint8 = imf.get_filtered([['same', dict(normalize=True, cast_dtype='uint8')]])
        np.testing.assert_allclose(result_float, [0, 1])
        np.testing.assert_array_equal(result_uint8, [0, 255])

    def test_normalization_constant(self):
        print('Testing normalization of a constant image ...')
        from squirrel.library.filters import ImageFilter
        img = np.full((3, 3), 8)
        result = ImageFilter(img).get_filtered([['same', dict(normalize=True)]])
        np.testing.assert_array_equal(result, 0)

    def test_mask_excluded_from_normalization(self):
        print('Testing zero mask exclusion from normalization ...')
        from squirrel.library.filters import ImageFilter
        img = np.array([0., 10., 20.])
        result = ImageFilter(img).get_filtered([['same', dict(keep_zeros=True, normalize=True)]])
        np.testing.assert_allclose(result, [0, 0, 1])

    def test_subtract_preserves_sign(self):
        print('Testing signed arithmetic in subtract mode ...')
        from squirrel.library.filters import ImageFilter
        img = np.array([10, 20], dtype=np.uint16)
        result = ImageFilter(img).get_filtered([['same', dict(filter_mode='subtract')]])
        np.testing.assert_array_equal(result, [0, 0])
        self.assertEqual(result.dtype.kind, 'f')

    def test_safe_cast(self):
        print('Testing safe integer casting ...')
        from squirrel.library.filters import ImageFilter
        img = np.array([-10., 100., 70000.])
        result = ImageFilter(img).get_filtered([['same', dict(cast_dtype='uint16')]])
        np.testing.assert_array_equal(result, [0, 100, 65535])

    def test_invalid_range(self):
        print('Testing invalid normalization ranges ...')
        from squirrel.library.filters import ImageFilter
        img = np.array([1., 2.])
        imf = ImageFilter(img)
        with self.assertRaises(ValueError):
            imf.get_filtered([['same', dict(normalize=True, normalization_range=(2, 2))]])
        with self.assertRaises(ValueError):
            imf.get_filtered([['same', dict(normalization_range=(0, 2))]])

    def test_invert_float(self):
        print('Testing inversion of floating-point images ...')
        from squirrel.library.filters import ImageFilter
        img = np.array([0.1, 0.8])
        result = ImageFilter.invert(img, invert_range=(0, 1))
        np.testing.assert_allclose(result, [0.9, 0.2])

    def test_median_seed(self):
        print('Testing reproducible median footprint sampling ...')
        from squirrel.library.filters import ImageFilter
        img = np.arange(100).reshape(10, 10)
        kwargs = dict(radius=2, sample_footprint=8, random_seed=42)
        np.testing.assert_array_equal(ImageFilter.median(img, **kwargs), ImageFilter.median(img, **kwargs))

    def test_median_binned_shape(self):
        print('Testing binned median reconstruction shape ...')
        from unittest.mock import patch
        from types import ModuleType
        import sys
        from squirrel.library.filters import ImageFilter
        img = np.arange(35).reshape(5, 7)
        scaling = ModuleType('squirrel.library.scaling')
        scaling.average_bin_image = lambda image, factor: image[::factor, ::factor]
        with patch.dict(sys.modules, {'squirrel.library.scaling': scaling}):
            result = ImageFilter.median(img, radius=1, on_binned=2)
        self.assertEqual(result.shape, img.shape)


class TestFilters(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_fft_constant(self):
        print('Testing fft_highpass on a constant image ...')
        from squirrel.library.filters import fft_highpass
        img = np.ones((12, 15))
        result = fft_highpass(img, (2, 3))
        self.assertLess(np.max(np.abs(result)), 1e-12)


if __name__ == '__main__':
    unittest.main()

class TestPipelineValidation(unittest.TestCase):

    def test_valid_pipeline(self):
        print('Testing valid filter pipeline ...')
        from squirrel.library.filters import ImageFilter
        steps = ImageFilter.validate_pipeline([
            ['gaussian', dict(sigma=2)],
            ['median', dict(radius=3, filter_mode='subtract')],
            ['same', dict(normalize=True, cast_dtype='uint16')]
        ])
        self.assertEqual(len(steps), 3)

    def test_invalid_step_structure(self):
        print('Testing invalid pipeline step structure ...')
        from squirrel.library.filters import ImageFilter
        for steps in ([['same']], [['same', {}, 'extra']], [['same', None]]):
            with self.assertRaises((TypeError, ValueError)):
                ImageFilter.validate_pipeline(steps)

    def test_unknown_filter(self):
        print('Testing unknown filter names ...')
        from squirrel.library.filters import ImageFilter
        with self.assertRaisesRegex(ValueError, 'step 1.*unknown filter'):
            ImageFilter.validate_pipeline([['same', {}], ['not_a_filter', {}]])

    def test_unknown_parameter(self):
        print('Testing unknown filter parameters ...')
        from squirrel.library.filters import ImageFilter
        with self.assertRaisesRegex(ValueError, 'step 0.*sigam'):
            ImageFilter.validate_pipeline([['gaussian', dict(sigam=2)]])

    def test_invalid_wrapper_options(self):
        print('Testing invalid filter wrapper options ...')
        from squirrel.library.filters import ImageFilter
        for kwargs in (dict(filter_mode='multiply'),
                       dict(normalization_range=(0, 1)),
                       dict(normalize=True, normalization_range=(2, 1)),
                       dict(cast_dtype='not_a_dtype')):
            with self.assertRaises(ValueError):
                ImageFilter.validate_pipeline([['same', kwargs]])

    def test_validation_before_execution(self):
        print('Testing validation before pipeline execution ...')
        from squirrel.library.filters import ImageFilter
        img = np.arange(9).reshape(3, 3)
        with self.assertRaisesRegex(ValueError, 'step 1.*sigam'):
            ImageFilter(img).get_filtered([['same', {}], ['gaussian', dict(sigam=2)]])

    def test_valid_pipeline_execution(self):
        print('Testing execution of validated pipeline ...')
        from squirrel.library.filters import ImageFilter
        img = np.arange(9).reshape(3, 3)
        result = ImageFilter(img).get_filtered([['same', dict(cast_dtype='float32')]])
        np.testing.assert_array_equal(result, img)
        self.assertEqual(result.dtype, np.float32)


class TestFilterSpecificValidation(unittest.TestCase):

    def test_invalid_gaussian_sigma(self):
        print('Testing invalid Gaussian sigma values ...')
        from squirrel.library.filters import ImageFilter
        for sigma in (0, -1, float('nan'), float('inf'), (1, -2), (1, 2, 3)):
            with self.subTest(sigma=sigma), self.assertRaisesRegex(ValueError, "step 0.*sigma"):
                ImageFilter.validate_pipeline([['gaussian', dict(sigma=sigma)]])

    def test_invalid_fft_sigma(self):
        print('Testing invalid FFT highpass sigma values ...')
        from squirrel.library.filters import ImageFilter
        for sigma in (0, (-1, 2), (1, 2, 3)):
            with self.subTest(sigma=sigma), self.assertRaisesRegex(ValueError, "step 0.*sigma"):
                ImageFilter.validate_pipeline([['fft_highpass', dict(sigma=sigma)]])

    def test_invalid_median_parameters(self):
        print('Testing invalid median parameters ...')
        from squirrel.library.filters import ImageFilter
        for kwargs in (dict(), dict(radius=1, footprint=np.ones((3, 3))),
                       dict(radius=-1), dict(radius=1.5), dict(radius=(1, -2)),
                       dict(footprint=np.zeros((3, 3))), dict(radius=1, on_binned=0),
                       dict(radius=1, sample_footprint=-1), dict(radius=1, random_seed=-1)):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'step 0'):
                ImageFilter.validate_pipeline([['median', kwargs]])

    def test_invalid_invert_range(self):
        print('Testing invalid inversion range ...')
        from squirrel.library.filters import ImageFilter
        with self.assertRaisesRegex(ValueError, 'step 0.*invert_range'):
            ImageFilter.validate_pipeline([['invert', dict(invert_range=(2, 1))]])

    def test_valid_filter_specific_parameters(self):
        print('Testing valid filter-specific parameters ...')
        from squirrel.library.filters import ImageFilter
        steps = ImageFilter.validate_pipeline([
            ['gaussian', dict(sigma=1.5)],
            ['fft_highpass', dict(sigma=(2, 3))],
            ['median', dict(radius=(1, 2), on_binned=2, sample_footprint=3, random_seed=42)],
            ['invert', dict(invert_range=(0, 1))],
        ])
        self.assertEqual(len(steps), 4)
