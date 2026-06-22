import unittest
import warnings
import numpy as np


class TestImageFilter(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_gaussian(self):
        print(f'Testing Gaussian filter ...')
        from squirrel.library.filters import ImageFilter
        from vigra.filters import gaussianSmoothing
        from skimage.data import camera
        img = camera()

        gaussian_ref = gaussianSmoothing(img, 2)
        imf = ImageFilter(img)
        gaussian = imf.get_filtered([['gaussian', dict(sigma=2)]])

        np.testing.assert_equal(gaussian_ref, gaussian)

    def test_gaussian_gradient_magnitude(self):
        print(f'Testing Gaussian gradient magnitude ...')
        from squirrel.library.filters import ImageFilter
        from vigra.filters import gaussianGradientMagnitude
        from skimage.data import camera
        img = camera()

        gaussian_ref = gaussianGradientMagnitude(img.astype('float32'), 2)
        imf = ImageFilter(img)
        gaussian = imf.get_filtered([['gaussian_gradient_magnitude', dict(sigma=2)]])

        np.testing.assert_equal(gaussian_ref, gaussian)

    def test_clahe(self):
        print(f'Testing CLAHE filter ...')
        from squirrel.library.filters import ImageFilter
        from cv2 import createCLAHE
        from skimage.data import camera
        img = camera()

        clip_limit = 3.0
        tile_grid_size = (127, 127)
        clahe = createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        clahe_filtered_ref = clahe.apply(img)
        imf = ImageFilter(img)
        clahe_filtered = imf.get_filtered([['clahe', dict()]])

        np.testing.assert_equal(clahe_filtered_ref, clahe_filtered)

    def test_vsnr(self):
        print(f'Testing VSNR ...')
        from squirrel.library.filters import ImageFilter
        from pyvsnr import vsnr2d
        from skimage.data import camera
        img = camera()

        filters = [dict(name='Gabor', noise_level=0.35, sigma=[1, 50], theta=90)]
        vsnr_ref = vsnr2d(img, filters)
        imf = ImageFilter(img)
        vsnr = imf.get_filtered([['vsnr', dict(filters=dict(name='Gabor', noise_level=0.35, sigma=[1, 50], theta=90))]])

        np.testing.assert_equal(vsnr_ref, vsnr)

    def test_same(self):
        print(f'Testing "same" filter ...')
        from squirrel.library.filters import ImageFilter
        from skimage.data import camera
        img = camera()

        imf = ImageFilter(img)
        same = imf.get_filtered([['same', dict()]])

        np.testing.assert_equal(img, same)

    def test_invert(self):
        print(f'Testing "invert" filter ...')
        from squirrel.library.filters import ImageFilter
        from skimage.data import camera
        img = camera()

        imf = ImageFilter(img)
        inverted = imf.get_filtered([['invert', dict()]])

        np.testing.assert_equal(255 - img, inverted)

    def test_filter_wrapper_clip(self):
        print(f'Testing filter wrapper clipping ...')
        from squirrel.library.filters import ImageFilter
        from skimage.data import camera
        img = camera()

        imf = ImageFilter(img)
        filtered = imf.get_filtered([['same', dict(clip=[100, 200])]])
                                    
        np.testing.assert_equal(filtered, np.clip(img, 100, 200))

    def test_filter_wrapper_cast_dtype_float32(self):
        print(f'Testing filter wrapper dtype casting to float32 ...')
        from squirrel.library.filters import ImageFilter
        from skimage.data import camera
        img = camera()

        imf = ImageFilter(img)
        filtered = imf.get_filtered([['same', dict(cast_dtype=np.float32)]])

        np.testing.assert_equal(filtered, img.astype(np.float32))

    def test_filter_wrapper_cast_dtype_uint8(self):
        print(f'Testing filter wrapper dtype casting to uint8 ...')

        from squirrel.library.filters import ImageFilter
        img = np.arange(256).reshape(16, 16)
        imf = ImageFilter(img.astype(np.float64) + 100 * 100)
        filtered = imf.get_filtered([['same', dict(cast_dtype=np.uint8)]])

        np.testing.assert_equal(filtered, img)

    def test_filter_wrapper_filter_mode(self):
        print(f'Testing filter mode of filter wrapper ...')

        from squirrel.library.filters import ImageFilter
        from skimage.data import camera

        img = camera()

        imf = ImageFilter(img)
        filtered1 = imf.get_filtered([['same', dict(filter_mode='add')]])
        filtered2 = imf.get_filtered([['same', dict(filter_mode='subtract')]])

        np.testing.assert_equal(filtered1, 2 * img.astype(np.float64))
        np.testing.assert_equal(filtered2, np.zeros_like(img, dtype=np.float64))
        self.assertEqual(filtered1.dtype, np.float32)
        self.assertEqual(filtered2.dtype, np.float32)

    def test_filter_wrapper_keep_zeros(self):
        print(f'Testing keep_zeros option of filter wrapper ...')
        from squirrel.library.filters import ImageFilter

        img = np.zeros((16, 16), dtype=np.uint8)
        img[8, 8] = 255

        imf = ImageFilter(img)
        filtered1 = imf.get_filtered([['gaussian', dict(keep_zeros=True, sigma=5)]])
        filtered2 = imf.get_filtered([['gaussian', dict(keep_zeros=False, sigma=5)]])

        # With keep_zeros=True, the zero pixels should remain zero
        self.assertEqual(filtered1[7, 7], 0)
        self.assertGreater(filtered1[8, 8], 0)  

        # With keep_zeros=False, the zero pixels can be modified by the filter
        self.assertGreater(filtered2[7, 7], 0)
        self.assertEqual(filtered1[8, 8], filtered2[8, 8])  # The non-zero pixel should be the same in both cases


class TestFilters(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_shape_and_type(self):
        print('Testing fft_highpass: shape and type ...')
        from squirrel.library.filters import fft_highpass
        img = np.random.rand(64, 128)
        result = fft_highpass(img, sigma=[10., 10.])

        # Test shape
        assert result.shape == img.shape

        # Test real-valued (allow tiny numerical errors)
        assert np.allclose(result.imag, 0, atol=1e-12) or np.isrealobj(result)

    def test_uniform_image(self):
        print('Testing fft_highpass: uniform image')
        from squirrel.library.filters import fft_highpass

        img = np.ones((32, 32))
        result = fft_highpass(img, sigma=[10., 5.])

        # High-pass of a uniform image should be near zero
        assert np.allclose(result, 0, atol=1e-12)

    def test_anisotropy_effect(self):
        print('Testing fft_highpass: anisotropy')
        from squirrel.library.filters import fft_highpass

        # create a horizontal gradient
        ny, nx = 64, 128
        img = np.tile(np.linspace(0, 1, nx), (ny, 1))

        # Strong x smoothing
        result_x = fft_highpass(img, sigma=[2., 20.])
        # Strong y smoothing
        result_y = fft_highpass(img, sigma=[20., 2.])

        # Outputs should differ
        assert not np.allclose(result_x, result_y)

    def test_highpass_behavior(self):
        print('Testing fft_highpass: highpass behavior')
        from squirrel.library.filters import fft_highpass

        # Image with low-frequency pattern + high-frequency noise
        ny, nx = 64, 64
        x = np.linspace(0, 4 * np.pi, nx)
        y = np.linspace(0, 4 * np.pi, ny)
        X, Y = np.meshgrid(x, y)
        img = np.sin(X / 2) + 0.1 * np.random.rand(ny, nx)  # low + high freq

        result = fft_highpass(img, sigma=[5., 5.])

        # High-pass should remove most of the low-frequency component
        # Compute mean absolute value of low-pass vs high-pass
        from scipy.ndimage import gaussian_filter
        lowpass = gaussian_filter(img, sigma=5)

        assert np.mean(np.abs(result)) > np.mean(np.abs(lowpass - img)) * 0.5

