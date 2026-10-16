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
        result = ImageFilter(img).get_filtered([
            ['same', dict(keep_zeros=True, ignore_zeros=True, normalize=True)]
        ])
        np.testing.assert_allclose(result, [0, 0, 1])

        included = ImageFilter(img).get_filtered([
            ['same', dict(keep_zeros=True, ignore_zeros=False, normalize=True)]
        ])
        np.testing.assert_allclose(included, [0, 0.5, 1])

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


class TestFilterInputEdgeCases(unittest.TestCase):

    def test_empty_image_direct_and_pipeline(self):
        print('Testing empty filter input rejection ...')
        from squirrel.library.filters import ImageFilter
        image = np.empty((0, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Filter 'same': .*empty"):
            ImageFilter.same(image)
        with self.assertRaisesRegex(ValueError, r"Filter step 0 \('same'\): .*empty"):
            ImageFilter(image).get_filtered([['same', {}]])

    def test_unsupported_image_dtypes(self):
        print('Testing unsupported filter input dtypes ...')
        from squirrel.library.filters import ImageFilter
        for image in (np.array([['a']], dtype=object),
                      np.array([[1 + 2j]]), np.array([[True]])):
            with self.subTest(dtype=image.dtype):
                with self.assertRaisesRegex(TypeError, 'real numeric dtype'):
                    ImageFilter.same(image)
                with self.assertRaisesRegex(TypeError, 'step 0.*real numeric dtype'):
                    ImageFilter(image).get_filtered([['same', {}]])

    def test_invalid_stack_dimensions_and_empty_stack(self):
        print('Testing invalid and empty stack inputs ...')
        from squirrel.library.filters import ImageFilter
        for shape, message in (((3, 3), '3D or 4D'),
                               ((1, 1, 1, 1, 1), '3D or 4D'),
                               ((0, 3, 3), 'nonempty'),
                               ((2, 0, 3), 'nonempty')):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, message):
                    ImageFilter(np.zeros(shape)).get_filtered_stack([['same', {}]])


class TestFilterWorkflowArgumentErrors(unittest.TestCase):

    def test_invalid_volume_arguments_before_input_io(self):
        print('Testing invalid volume filtering arguments before input I/O ...')
        from squirrel.workflows.volume import filter_2d_workflow
        with patch('squirrel.library.io.load_data_handle') as load_handle, \
             patch('squirrel.library.io.get_filetype', return_value='dir'):
            for kwargs in (dict(batch_size=0), dict(batch_size=True),
                           dict(batch_size=1.5), dict(n_workers=0),
                           dict(n_workers=True), dict(n_workers='2')):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'batch_size|n_workers'):
                    filter_2d_workflow('input', 'output', filters=[['same', {}]], **kwargs)
            load_handle.assert_not_called()

    def test_unsupported_batched_output_before_input_io(self):
        print('Testing unsupported batched output is rejected before input I/O ...')
        from squirrel.workflows.volume import filter_2d_workflow
        with patch('squirrel.library.io.load_data_handle') as load_handle, \
             patch('squirrel.library.io.get_filetype', return_value='h5'):
            with self.assertRaisesRegex(ValueError, 'directory or OME-Zarr'):
                filter_2d_workflow('input', 'output.h5', filters=[['same', {}]], batch_size=2)
            load_handle.assert_not_called()

    def test_empty_unbatched_stack_before_output_io(self):
        print('Testing empty unbatched volume is rejected before output I/O ...')
        from squirrel.workflows.volume import filter_2d_workflow
        stack = np.empty((0, 3, 3), dtype=np.uint8)
        with patch('squirrel.library.io.load_data_handle', return_value=(stack, stack.shape)), \
             patch('squirrel.library.io.write_stack') as write_stack:
            with self.assertRaisesRegex(ValueError, 'empty stack'):
                filter_2d_workflow('input', 'output', filters=[['same', {}]])
            write_stack.assert_not_called()

    def test_constant_intermediate_preview(self):
        print('Testing constant intermediate previews avoid division by zero ...')
        from squirrel.workflows.image import filter_2d_workflow
        image = np.ones((3, 3), dtype=np.uint8)
        with patch('squirrel.library.io.read_tif_slice', return_value=image), \
             patch('squirrel.library.io.write_tif_slice') as write_slice, \
             patch('squirrel.workflows.image.os.makedirs'):
            filter_2d_workflow('input.tif', 'output.tif', filters=[['same', {}]],
                               save_intermediates=True)
        self.assertGreaterEqual(write_slice.call_count, 2)
        preview = write_slice.call_args_list[0].args[0]
        self.assertEqual(preview.dtype, np.uint8)
        np.testing.assert_array_equal(preview, 0)


class TestSharedBooleanOptionValidation(unittest.TestCase):

    def test_numpy_boolean_options_direct_and_pipeline(self):
        print('Testing shared boolean validation accepts NumPy booleans ...')
        from squirrel.library.filters import ImageFilter
        image = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        for key in ('normalize', 'keep_zeros', 'zeros_to_mean'):
            with self.subTest(key=key):
                options = {key: np.bool_(False)}
                ImageFilter.validate_pipeline([['same', options]])
                np.testing.assert_array_equal(ImageFilter.same(image, **options), image)

    def test_boolean_option_errors_direct_and_pipeline(self):
        print('Testing shared boolean validation rejects non-boolean values ...')
        from squirrel.library.filters import ImageFilter
        image = np.array([[1, 2]], dtype=np.uint8)
        for key in ('normalize', 'keep_zeros', 'zeros_to_mean'):
            with self.subTest(key=key):
                with self.assertRaisesRegex(TypeError, key):
                    ImageFilter.validate_pipeline([['same', {key: 1}]])
                with self.assertRaisesRegex(TypeError, key):
                    ImageFilter.same(image, **{key: 1})


class TestFilterExecutionOptions(unittest.TestCase):

    def test_invalid_get_filtered_flags(self):
        print('Testing get_filtered rejects non-boolean execution options ...')
        from squirrel.library.filters import ImageFilter
        image_filter = ImageFilter(np.arange(4).reshape(2, 2))
        for key in ('return_intermediates', 'verbose'):
            for value in (0, 1, 'false', None):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(TypeError, key):
                        image_filter.get_filtered([['same', {}]], **{key: value})

    def test_numpy_boolean_get_filtered_flags(self):
        print('Testing get_filtered accepts NumPy boolean execution options ...')
        from squirrel.library.filters import ImageFilter
        image = np.arange(4).reshape(2, 2)
        result, intermediates = ImageFilter(image).get_filtered(
            [['same', {}]], return_intermediates=np.bool_(True), verbose=np.bool_(False))
        np.testing.assert_array_equal(result, image)
        self.assertEqual(len(intermediates), 1)
        np.testing.assert_array_equal(intermediates[0], image)


class TestFilterMultiprocessing(unittest.TestCase):

    def test_multiprocessing_consistency(self):
        print('Testing multiprocessing produces the same filtered stack as serial execution ...')
        from squirrel.library.filters import ImageFilter

        rng = np.random.default_rng(42)
        stack = rng.integers(0, 65535, size=(8, 64, 64), dtype=np.uint16)
        pipeline = [['fft_highpass', dict(sigma=(5, 5))]]
        serial = ImageFilter(stack).get_filtered_stack(pipeline, n_workers=1)
        parallel = ImageFilter(stack).get_filtered_stack(pipeline, n_workers=2)
        np.testing.assert_allclose(parallel, serial, rtol=1e-6, atol=1e-6)
        self.assertEqual(parallel.dtype, serial.dtype)

    def test_execution_backends(self):
        print('Testing serial, multiprocessing, and threading backend equivalence ...')
        from squirrel.library.filters import ImageFilter

        rng = np.random.default_rng(123)
        stack = rng.integers(0, 65535, size=(8, 64, 64), dtype=np.uint16)
        pipeline = [['fft_highpass', dict(sigma=(5, 5))]]
        reference = ImageFilter(stack).get_filtered_stack(
            pipeline, n_workers=1, backend='serial')
        for backend, workers in (('multiprocessing', 1),
                                 ('multiprocessing', 2),
                                 ('threading', 1),
                                 ('threading', 2),
                                 ('threading', 4)):
            with self.subTest(backend=backend, workers=workers):
                result = ImageFilter(stack).get_filtered_stack(
                    pipeline, n_workers=workers, backend=backend)
                np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-6)
                self.assertEqual(result.dtype, reference.dtype)
                self.assertEqual(result.shape, reference.shape)

    def test_default_backend_uses_threading(self):
        print('Testing threading is the default parallel stack backend ...')
        from unittest.mock import patch
        from squirrel.library.filters import ImageFilter

        stack = np.arange(4 * 8 * 8, dtype=np.uint16).reshape(4, 8, 8)
        pipeline = [['same', {}]]
        with patch('squirrel.library.filters.Pool', side_effect=AssertionError(
                'Default execution unexpectedly created a process pool')):
            result = ImageFilter(stack).get_filtered_stack(pipeline, n_workers=2)
        np.testing.assert_array_equal(result, stack)

    def test_mixed_pipeline_backend_equivalence(self):
        print('Testing mixed filter pipelines across execution backends ...')
        from squirrel.library.filters import ImageFilter

        rng = np.random.default_rng(230)
        stack = rng.integers(0, 65535, size=(6, 48, 48), dtype=np.uint16)
        pipeline = [
            ['gaussian', {'sigma': 1.5}],
            ['median', {'radius': 2}],
            ['fft_highpass', {'sigma': (5, 5)}],
        ]
        reference = ImageFilter(stack).get_filtered_stack(
            pipeline, n_workers=1, backend='serial')
        for backend in ('threading', 'multiprocessing'):
            with self.subTest(backend=backend):
                result = ImageFilter(stack).get_filtered_stack(
                    pipeline, n_workers=2, backend=backend)
                np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-6)
                self.assertEqual(result.dtype, reference.dtype)
                self.assertEqual(result.shape, reference.shape)

    def test_thread_worker_exception_propagation(self):
        print('Testing thread worker exceptions propagate to the caller ...')
        from squirrel.library.filters import ImageFilter

        from unittest.mock import patch
        stack = np.ones((4, 8, 8), dtype=np.uint8)

        def fail_on_slice(image, filters):
            raise RuntimeError('worker failure sentinel')

        with patch('squirrel.library.filters._filter_slice', side_effect=fail_on_slice):
            with self.assertRaisesRegex(RuntimeError, 'worker failure sentinel'):
                ImageFilter(stack).get_filtered_stack(
                    [['same', {}]], n_workers=2, backend='threading')

    def test_clahe_multiprocessing_after_serial(self):
        print('Testing CLAHE multiprocessing after serial OpenCV initialization ...')
        from squirrel.library.filters import ImageFilter

        rng = np.random.default_rng(2026)
        stack = rng.integers(0, 65535, size=(4, 64, 64), dtype=np.uint16)
        pipeline = [['clahe', {'clip_limit': 3.0, 'tile_grid_size': (8, 8)}]]
        reference = ImageFilter(stack).get_filtered_stack(
            pipeline, n_workers=1, backend='serial')
        result = ImageFilter(stack).get_filtered_stack(
            pipeline, n_workers=2, backend='multiprocessing')
        np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-6)

    def test_invalid_execution_backends(self):
        print('Testing invalid filtering execution backend options ...')
        from squirrel.library.filters import ImageFilter

        stack = np.ones((2, 4, 4), dtype=np.uint8)
        for backend in ('threads', '', 'invalid'):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, 'backend'):
                    ImageFilter(stack).get_filtered_stack(
                        [['same', {}]], backend=backend)
        for backend in (None, 1, True):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(TypeError, 'backend'):
                    ImageFilter(stack).get_filtered_stack(
                        [['same', {}]], backend=backend)
        with self.assertRaisesRegex(ValueError, 'n_workers'):
            ImageFilter(stack).get_filtered_stack(
                [['same', {}]], n_workers=2, backend='serial')

    @unittest.skipUnless(
        __import__('os').environ.get('SQUIRREL_RUN_FILTER_BENCHMARK') == '1',
        'Opt-in benchmark: set SQUIRREL_RUN_FILTER_BENCHMARK=1',
    )
    def test_multiprocessing_performance(self):
        print('Benchmarking multiprocessing across image sizes (informational only) ...')
        import statistics
        import time
        from squirrel.library.filters import ImageFilter

        pipeline = [['fft_highpass', dict(sigma=(5, 5))]]
        image_sizes = (256, 512, 1024)
        n_slices = 32
        repeats = 3
        worker_counts = (1, 2, 4)
        rng = np.random.default_rng(42)

        print(f'  Stack depth: {n_slices}; repetitions: {repeats}; '
              f'worker counts: {worker_counts}')
        for size in image_sizes:
            # Allocate only one stack size at a time to limit peak memory use.
            stack = rng.integers(0, 65535,
                                 size=(n_slices, size, size), dtype=np.uint16)
            times = {}
            print(f'\n  Image size: {size} x {size} ({n_slices} slices)')
            for workers in worker_counts:
                # Warm up imports, FFT machinery, and process startup.
                ImageFilter(stack[:2]).get_filtered_stack(pipeline, n_workers=workers)
                samples = []
                for run_index in range(filter_repeats):
                    print(f'    timing: {label}, run {run_index + 1}/{filter_repeats} ...', flush=True)
                    start = time.perf_counter()
                    result = ImageFilter(stack).get_filtered_stack(
                        pipeline, n_workers=workers)
                    samples.append(time.perf_counter() - start)
                    del result
                times[workers] = statistics.median(samples)
                print(f'    {workers} worker(s): {times[workers]:.3f} s median '
                      f'(runs: {", ".join(f"{t:.3f}" for t in samples)})')

            for workers in worker_counts[1:]:
                print(f'    Speedup with {workers} workers: '
                      f'{times[1] / times[workers]:.2f}x')
            del stack
        # No speedup assertion: timings vary with machine load and hardware.

    @unittest.skipUnless(
        __import__('os').environ.get('SQUIRREL_RUN_FILTER_BENCHMARK') == '1',
        'Opt-in benchmark: set SQUIRREL_RUN_FILTER_BENCHMARK=1',
    )
    def test_multiprocessing_chunking_and_threads(self):
        print('Benchmarking process chunk sizes and threads (informational only) ...')
        import statistics
        import time
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial
        from multiprocessing import Pool
        from squirrel.library.filters import ImageFilter, _filter_slice

        pipeline = [['fft_highpass', dict(sigma=(5, 5))]]
        sizes = (512, 1024)
        n_slices = 32
        repeats = 3
        rng = np.random.default_rng(42)
        worker = partial(_filter_slice, filters=pipeline)

        def serial(stack):
            return ImageFilter(stack).get_filtered_stack(pipeline, n_workers=1)

        def processes(stack, n_workers, chunksize):
            with Pool(processes=n_workers) as pool:
                return np.asarray(list(pool.imap(worker, stack, chunksize=chunksize)))

        def threads(stack, n_workers):
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                return np.asarray(list(pool.map(worker, stack)))

        for size in sizes:
            # Allocate only one stack at a time; discard outputs after each run.
            stack = rng.integers(0, 65535, size=(n_slices, size, size), dtype=np.uint16)
            print(f'\n  Image size: {size} x {size} ({n_slices} slices)')
            configurations = [('serial', serial)]
            configurations += [
                (f'processes: 4 workers, chunksize={chunk}',
                 partial(processes, n_workers=4, chunksize=chunk))
                for chunk in (1, 2, 4, 8)
            ]
            configurations += [
                (f'threads: {workers} workers', partial(threads, n_workers=workers))
                for workers in (2, 4)
            ]

            # Verify equivalence before measuring; do not mix correctness checks
            # into timed sections. This also catches ordering/dtype regressions.
            expected = serial(stack)
            for label, run in configurations[1:]:
                actual = run(stack)
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6,
                                           err_msg=label)
                self.assertEqual(actual.dtype, expected.dtype, label)
                del actual
            del expected

            timings = {}
            for label, run in configurations:
                # Each timed call includes pool creation, IPC, and output assembly.
                warmup = run(stack[:2])
                del warmup
                samples = []
                for run_index in range(filter_repeats):
                    print(f'    timing: {label}, run {run_index + 1}/{filter_repeats} ...', flush=True)
                    start = time.perf_counter()
                    result = run(stack)
                    samples.append(time.perf_counter() - start)
                    del result
                timings[label] = statistics.median(samples)
                print(f'    {label}: {timings[label]:.3f} s median '
                      f'(runs: {", ".join(f"{t:.3f}" for t in samples)})')

            baseline = timings['serial']
            for label in list(timings)[1:]:
                print(f'    Speedup ({label}): {baseline / timings[label]:.2f}x')
            del stack
        # No speedup assertion: scheduling, contention, and hardware vary.


    @unittest.skipUnless(
        __import__('os').environ.get('SQUIRREL_RUN_FILTER_BENCHMARK') == '1',
        'Opt-in benchmark: set SQUIRREL_RUN_FILTER_BENCHMARK=1',
    )
    def test_filter_backend_comparison(self):
        print('Benchmarking filtering backends across filter types (informational only) ...')
        import statistics
        import time
        from squirrel.library.filters import ImageFilter

        import cv2
        cv2.setNumThreads(1)

        # Keep inputs deterministic and identical for each execution backend.
        rng = np.random.default_rng(2026)
        n_slices = 16
        size = 512
        repeats = 3
        stack = rng.integers(0, 65535, size=(n_slices, size, size), dtype=np.uint16)
        pipelines = (
            ('gaussian', [['gaussian', {'sigma': 2.0}]]),
            ('median', [['median', {'radius': 2}]]),
            ('clahe', [['clahe', {'clip_limit': 3.0,
                                   'tile_grid_size': (8, 8)}]]),
            ('fft_highpass', [['fft_highpass', {'sigma': (5, 5)}]]),
        )
        configurations = (
            ('serial', 1, 'serial'),
            ('processes: 4', 4, 'multiprocessing'),
            ('threads: 2', 2, 'threading'),
            ('threads: 4', 4, 'threading'),
        )
        print(f'  Stack: {n_slices} x {size} x {size}; repetitions: {repeats}')

        for filter_name, pipeline in pipelines:
            # CLAHE is substantially more expensive than the other filters.
            # Use a smaller stack and fewer repetitions for this case.
            if filter_name == 'clahe':
                filter_stack = stack[:4, :128, :128]
                filter_repeats = 1
            else:
                filter_stack = stack
                filter_repeats = repeats
            print(f'\n  Filter: {filter_name} '
                  f'({filter_stack.shape[0]} x {filter_stack.shape[1]} x '
                  f'{filter_stack.shape[2]}; repetitions: {filter_repeats})')
            image_filter = ImageFilter(filter_stack)

            def run(n_workers, backend):
                return image_filter.get_filtered_stack(
                    pipeline, n_workers=n_workers, backend=backend)

            # Validate outputs before timing; timing excludes comparisons.
            print('    correctness: serial reference ...', flush=True)
            reference = run(1, 'serial')
            print('    correctness: serial reference complete', flush=True)
            for label, n_workers, backend in configurations[1:]:
                with self.subTest(filter=filter_name, backend=label):
                    print(f'    correctness: {label} ...', flush=True)
                    actual = run(n_workers, backend)
                    print(f'    correctness: {label} complete', flush=True)
                    np.testing.assert_allclose(
                        actual, reference, rtol=1e-6, atol=1e-6,
                        err_msg=f'{filter_name}: {label}')
                    self.assertEqual(actual.dtype, reference.dtype)
                    self.assertEqual(actual.shape, reference.shape)
                    del actual
            del reference

            timings = {}
            for label, n_workers, backend in configurations:
                # Pool creation and output assembly are part of the measurement.
                print(f'    warm-up: {label} ...', flush=True)
                warmup = run(n_workers, backend)
                print(f'    warm-up: {label} complete', flush=True)
                del warmup
                samples = []
                for run_index in range(filter_repeats):
                    print(f'    timing: {label}, run {run_index + 1}/{filter_repeats} ...', flush=True)
                    start = time.perf_counter()
                    result = run(n_workers, backend)
                    samples.append(time.perf_counter() - start)
                    del result
                timings[label] = statistics.median(samples)
                print(f'    {label}: {timings[label]:.3f} s median '
                      f'(runs: {", ".join(f"{t:.3f}" for t in samples)})')

            baseline = timings['serial']
            for label in list(timings)[1:]:
                print(f'    Speedup ({label}): {baseline / timings[label]:.2f}x')
        # No performance assertions: execution times are hardware-dependent.


class TestRescaleDtype(unittest.TestCase):

    def setUp(self):
        import warnings
        warnings.simplefilter('ignore', category=Warning)
        from squirrel.library.filters import ImageFilter
        self.ImageFilter = ImageFilter

    def test_rescale_dtype_fixed_uint16_to_uint8(self):
        print('Testing rescale_dtype: fixed full-range mapping uint16 to uint8')
        image = np.array([[0, 1000, 32768, 65535]], dtype=np.uint16)
        result = self.ImageFilter(image).get_filtered([['same', {'rescale_dtype': 'uint8'}]])
        expected = np.array([[0, 4, 128, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_rescale_dtype_fixed_uint8_to_uint16(self):
        print('Testing rescale_dtype: fixed full-range mapping uint8 to uint16')
        image = np.array([[0, 1, 128, 255]], dtype=np.uint8)
        result = self.ImageFilter(image).get_filtered([['same', {'rescale_dtype': 'uint16'}]])
        np.testing.assert_array_equal(result, image.astype(np.uint16) * 257)

    def test_rescale_dtype_independent_of_slice_range(self):
        print('Testing rescale_dtype: same input intensity maps identically in different slices')
        stack = np.array([[[0, 32768]], [[32768, 65535]]], dtype=np.uint16)
        result = self.ImageFilter(stack).get_filtered_stack([['same', {'rescale_dtype': 'uint8'}]],
                                                       progress=False)
        self.assertEqual(result[0, 0, 1], result[1, 0, 0])
        self.assertEqual(result[0, 0, 1], 128)

    def test_rescale_dtype_rejects_conflicts_and_invalid_types(self):
        print('Testing rescale_dtype: actionable errors for conflicting options and types')
        image = np.array([[0, 65535]], dtype=np.uint16)
        for params, expected in (
                ({'rescale_dtype': 'uint8', 'normalize': True}, 'normalize=True'),
                ({'rescale_dtype': 'uint8', 'cast_dtype': 'uint8'}, 'cast_dtype'),
                ({'rescale_dtype': 'float32'}, 'uint8 or uint16')):
            with self.subTest(params=params):
                with self.assertRaisesRegex((TypeError, ValueError), expected):
                    self.ImageFilter(image).get_filtered([['same', params]])
        for dtype in (np.float32, np.int16, np.uint32):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(TypeError, 'input/output'):
                    self.ImageFilter(image.astype(dtype)).get_filtered(
                        [['same', {'rescale_dtype': 'uint8'}]])
