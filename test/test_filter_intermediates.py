import os
import tempfile
import unittest
import warnings

import numpy as np


class TestFilterIntermediates(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)
        self.stack = np.arange(48, dtype=np.uint8).reshape(3, 4, 4)
        self.pipeline = [['same', {}], ['same', {}]]

    def test_stack_intermediates_backends(self):
        print('Testing intermediate stack results for all backends')
        from squirrel.library.filters import ImageFilter

        for backend, workers in [('serial', 1), ('threading', 2), ('multiprocessing', 2)]:
            with self.subTest(backend=backend):
                result, steps = ImageFilter(self.stack).get_filtered_stack(
                    self.pipeline, backend=backend, n_workers=workers,
                    return_intermediates=True, progress=False)
                np.testing.assert_array_equal(result, self.stack)
                self.assertEqual(len(steps), 2)
                for step in steps:
                    np.testing.assert_array_equal(step, self.stack)

    def test_volume_intermediate_previews(self):
        print('Testing volume intermediate previews with batching')
        from squirrel.workflows.volume import filter_2d_workflow
        from squirrel.library.io import write_tif_stack, read_tif_slice

        for batch_size in (None, 2):
            with self.subTest(batch_size=batch_size), tempfile.TemporaryDirectory() as tmp:
                source = os.path.join(tmp, 'input')
                output = os.path.join(tmp, 'output_stack')
                write_tif_stack(self.stack, source, slice_name='slice_{:05d}.tif')
                filter_2d_workflow(source, output, filters=self.pipeline,
                                   batch_size=batch_size, progress=False,
                                   save_intermediates=True)
                for index in range(3):
                    name = f'slice_{index:05d}.tif'
                    np.testing.assert_array_equal(
                        read_tif_slice(os.path.join(output,
                            f'slice_{index:04d}.tif' if batch_size is None else name), False),
                        self.stack[index])
                    for step in range(2):
                        filepath = os.path.join(
                            tmp, 'output_stack_intermediates', f'filter{step:02d}_same', name)
                        preview = read_tif_slice(filepath, False)
                        self.assertEqual(preview.dtype, np.uint8)
                        self.assertEqual(preview.min(), 0)
                        self.assertEqual(preview.max(), 255)

    def test_volume_default_no_intermediates(self):
        print('Testing disabled intermediate output')
        from squirrel.workflows.volume import filter_2d_workflow
        from squirrel.library.io import write_tif_stack

        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, 'input')
            output = os.path.join(tmp, 'output_stack')
            write_tif_stack(self.stack, source, slice_name='slice_{:05d}.tif')
            filter_2d_workflow(source, output, filters=self.pipeline, progress=False)
            self.assertFalse(os.path.exists(os.path.join(tmp, 'output_stack_intermediates')))
