import unittest
import warnings
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch
import numpy as np




class TestFilterWorkflowValidation(unittest.TestCase):

    def test_image_invalid_pipeline_before_io(self):
        print('Testing image workflow validation before I/O ...')
        from squirrel.workflows.image import filter_2d_workflow
        with patch('squirrel.library.io.read_tif_slice') as read_slice:
            with self.assertRaisesRegex(ValueError, 'unknown filter'):
                filter_2d_workflow('input.tif', 'output.tif',
                                   filters=[['unknown_filter', {}]])
            read_slice.assert_not_called()

    def test_volume_invalid_pipeline_before_io(self):
        print('Testing volume workflow validation before I/O ...')
        from squirrel.workflows.volume import filter_2d_workflow
        with patch('squirrel.library.io.load_data_handle') as load_handle:
            with self.assertRaisesRegex(ValueError, 'unknown filter'):
                filter_2d_workflow('input', 'output.ome.zarr',
                                   filters=[['unknown_filter', {}]], batch_size=8)
            load_handle.assert_not_called()


class TestFilterWorkflowDtype(unittest.TestCase):

    def test_ome_zarr_uses_filtered_dtype(self):
        print('Testing OME-Zarr output dtype follows filtering ...')
        from squirrel.workflows.volume import filter_2d_workflow

        stack = np.arange(24, dtype=np.uint16).reshape(3, 2, 4)
        store = MagicMock()
        zarr_module = ModuleType('squirrel.library.ome_zarr')
        zarr_module.OMEZarrStore = MagicMock()
        zarr_module.OMEZarrStore.create.return_value = store

        for pipeline, expected_dtype in (
            ([['same', dict(cast_dtype='uint8')]], np.dtype('uint8')),
            ([['same', dict(filter_mode='add')]], np.dtype('float32')),
            ([['same', dict(normalize=True)]], np.dtype('float32')),
        ):
            with self.subTest(pipeline=pipeline):
                zarr_module.OMEZarrStore.create.reset_mock()
                store.reset_mock()
                with patch('squirrel.library.io.load_data_handle', return_value=(stack, stack.shape)), \
                     patch('squirrel.library.io.get_filetype', return_value='ome_zarr'), \
                     patch.dict(sys.modules, {'squirrel.library.ome_zarr': zarr_module}):
                    filter_2d_workflow('input', 'output.ome.zarr', filters=pipeline,
                                       batch_size=2, n_workers=1)
                self.assertEqual(zarr_module.OMEZarrStore.create.call_args.kwargs['dtype'], expected_dtype)
                self.assertEqual(store.write.call_count, 2)
                for call in store.write.call_args_list:
                    self.assertEqual(call.kwargs['data'].dtype, expected_dtype)

    def test_invalid_batch_size_before_output_creation(self):
        print('Testing invalid batch size is rejected before output creation ...')
        from squirrel.workflows.volume import filter_2d_workflow
        stack = np.zeros((2, 2, 2), dtype=np.uint16)
        with patch('squirrel.library.io.load_data_handle', return_value=(stack, stack.shape)), \
             patch('squirrel.library.io.get_filetype', return_value='ome_zarr'):
            with self.assertRaisesRegex(ValueError, 'batch_size'):
                filter_2d_workflow('input', 'output.ome.zarr',
                                   filters=[['same', {}]], batch_size=0)


class TestFilterBooleanParameters(unittest.TestCase):

    def test_invalid_boolean_parameters_in_pipeline(self):
        print('Testing invalid boolean filter options in pipeline ...')
        from squirrel.library.filters import ImageFilter
        for key in ('normalize', 'keep_zeros', 'zeros_to_mean'):
            for value in (1, 0, 'true', None):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(TypeError, f'step 0.*{key}'):
                        ImageFilter.validate_pipeline([['same', {key: value}]])

    def test_invalid_boolean_parameters_direct_call(self):
        print('Testing invalid boolean filter options in direct calls ...')
        from squirrel.library.filters import ImageFilter
        for key in ('normalize', 'keep_zeros', 'zeros_to_mean'):
            with self.subTest(key=key):
                with self.assertRaisesRegex(TypeError, key):
                    ImageFilter.same(np.array([1, 2]), **{key: 'false'})

    def test_valid_boolean_parameters(self):
        print('Testing valid boolean filter options ...')
        from squirrel.library.filters import ImageFilter
        pipeline = [['same', dict(normalize=False, keep_zeros=True, zeros_to_mean=False)]]
        ImageFilter.validate_pipeline(pipeline)
        np.testing.assert_array_equal(ImageFilter(np.array([0, 1])).get_filtered(pipeline), [0, 1])


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


class TestFilterStackWorkers(unittest.TestCase):

    def test_invalid_worker_count(self):
        print('Testing invalid stack filter worker counts ...')
        from squirrel.library.filters import ImageFilter
        stack = np.zeros((2, 3, 4), dtype=np.uint8)
        for n_workers in (0, -1, 1.5, '2', True, None):
            with self.subTest(n_workers=n_workers):
                with self.assertRaisesRegex(ValueError, 'n_workers'):
                    ImageFilter(stack).get_filtered_stack([['same', {}]], n_workers=n_workers)

    def test_numpy_integer_worker_count(self):
        print('Testing NumPy integer stack filter worker count ...')
        from squirrel.library.filters import ImageFilter
        stack = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        result = ImageFilter(stack).get_filtered_stack([['same', {}]], n_workers=np.int64(1))
        np.testing.assert_array_equal(result, stack)


class TestFilterExceptionConsistency(unittest.TestCase):

    def test_range_types_in_pipeline_and_direct_calls(self):
        print('Testing range parameter type and value errors ...')
        from squirrel.library.filters import ImageFilter
        image = np.arange(4).reshape(2, 2)
        for value, exception in ((3, TypeError), ('0,1', TypeError),
                                 ([0, '1'], TypeError), ((0,), ValueError),
                                 ((2, 1), ValueError), ((0, float('nan')), ValueError)):
            with self.subTest(value=value):
                with self.assertRaisesRegex(exception, 'step 0.*clip'):
                    ImageFilter.validate_pipeline([['same', {'clip': value}]])
                with self.assertRaisesRegex(exception, 'clip'):
                    ImageFilter.same(image, clip=value)

    def test_filter_mode_type_and_value(self):
        print('Testing filter_mode type and value errors ...')
        from squirrel.library.filters import ImageFilter
        image = np.arange(4).reshape(2, 2)
        for value, exception in ((None, TypeError), (1, TypeError),
                                 (['apply'], TypeError), ('multiply', ValueError)):
            with self.subTest(value=value):
                with self.assertRaisesRegex(exception, 'step 0.*filter_mode'):
                    ImageFilter.validate_pipeline([['same', {'filter_mode': value}]])
                with self.assertRaisesRegex(exception, 'filter_mode'):
                    ImageFilter.same(image, filter_mode=value)

    def test_valid_numpy_range(self):
        print('Testing valid NumPy range parameters ...')
        from squirrel.library.filters import ImageFilter
        image = np.arange(4).reshape(2, 2)
        pipeline = [['same', {'clip': np.array([1, 2])}]]
        ImageFilter.validate_pipeline(pipeline)
        np.testing.assert_array_equal(ImageFilter(image).get_filtered(pipeline), [[1, 1], [2, 2]])


class TestDirectFilterParameterValidation(unittest.TestCase):

    def test_invalid_filter_parameters_direct_and_pipeline(self):
        print('Testing consistent filter-specific validation for direct calls and pipelines ...')
        from squirrel.library.filters import ImageFilter
        image = np.arange(25, dtype=np.uint8).reshape(5, 5)
        cases = (
            ('gaussian', {'sigma': -1}, 'sigma'),
            ('fft_highpass', {'sigma': (1, -2)}, 'sigma'),
            ('median', {'radius': -1}, 'radius'),
            ('median', {'radius': 1, 'on_binned': 0}, 'on_binned'),
            ('median', {'radius': 1, 'sample_footprint': -1}, 'sample_footprint'),
            ('invert', {'invert_range': (2, 1)}, 'invert_range'),
        )
        for name, parameters, message in cases:
            with self.subTest(name=name, parameters=parameters):
                with self.assertRaisesRegex(ValueError, message):
                    ImageFilter.validate_pipeline([[name, parameters]])
                with self.assertRaisesRegex(ValueError, message):
                    getattr(ImageFilter, name)(image, **parameters)

    def test_positional_arguments_are_validated(self):
        print('Testing direct positional filter argument validation ...')
        from squirrel.library.filters import ImageFilter
        image = np.ones((5, 5), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, 'radius'):
            ImageFilter.median(image, -1)
        with self.assertRaisesRegex(ValueError, 'sigma'):
            ImageFilter.fft_highpass(image, -1)

    def test_valid_direct_filter_call(self):
        print('Testing valid direct filter parameters ...')
        from squirrel.library.filters import ImageFilter
        image = np.arange(25, dtype=np.uint8).reshape(5, 5)
        np.testing.assert_array_equal(ImageFilter.same(image), image)
        np.testing.assert_array_equal(ImageFilter.invert(image, invert_range=(0, 255)), 255 - image)


class TestFilterErrorMessages(unittest.TestCase):

    def test_direct_validation_identifies_filter_and_parameter(self):
        print('Testing direct filter validation error context ...')
        from squirrel.library.filters import ImageFilter
        with self.assertRaisesRegex(ValueError, r"Filter 'median': .*radius") as caught:
            ImageFilter.median(np.ones((3, 3)), radius=-1)
        self.assertNotIn('step 0', str(caught.exception))

    def test_pipeline_execution_identifies_step_once(self):
        print('Testing pipeline execution error context ...')
        from squirrel.library.filters import ImageFilter
        image = np.ones((3, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, r"Filter step 1 \('fft_highpass'\): .*2D") as caught:
            ImageFilter(image[0]).get_filtered([['same', {}], ['fft_highpass', {}]])
        self.assertEqual(str(caught.exception).count("'fft_highpass'"), 1)

    def test_direct_execution_identifies_filter(self):
        print('Testing direct filter execution error context ...')
        from squirrel.library.filters import ImageFilter
        with self.assertRaisesRegex(ValueError, r"Filter 'fft_highpass': .*2D"):
            ImageFilter.fft_highpass(np.ones(5))
