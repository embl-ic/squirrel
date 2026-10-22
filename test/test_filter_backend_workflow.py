import unittest
import warnings
from unittest.mock import patch

import numpy as np


class TestFilterBackendWorkflow(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_backend_forwarded_to_filter_stack(self):
        print('Testing volume workflow forwards the selected backend')
        from squirrel.library.filters import ImageFilter
        from squirrel.workflows.volume import filter_2d_workflow

        stack = np.arange(24, dtype=np.uint8).reshape(3, 2, 4)
        for backend in ('threading', 'multiprocessing', 'serial'):
            with self.subTest(backend=backend):
                with patch('squirrel.library.io.load_data_handle', return_value=(stack, stack.shape)), \
                     patch('squirrel.library.io.write_stack') as write_stack, \
                     patch.object(ImageFilter, 'get_filtered_stack', return_value=stack) as get_filtered:
                    filter_2d_workflow('input', 'output', filters=[['same', {}]],
                                       n_workers=1, backend=backend, progress=False)
                self.assertEqual(get_filtered.call_args.kwargs['backend'], backend)
                self.assertEqual(get_filtered.call_args.kwargs['n_workers'], 1)
                np.testing.assert_array_equal(write_stack.call_args.args[1], stack)

    def test_default_backend(self):
        print('Testing default workflow backend remains threading')
        from squirrel.library.filters import ImageFilter
        from squirrel.workflows.volume import filter_2d_workflow

        stack = np.zeros((2, 3, 4), dtype=np.uint8)
        with patch('squirrel.library.io.load_data_handle', return_value=(stack, stack.shape)), \
             patch('squirrel.library.io.write_stack'), \
             patch.object(ImageFilter, 'get_filtered_stack', return_value=stack) as get_filtered:
            filter_2d_workflow('input', 'output', filters=[['same', {}]], progress=False)
        self.assertEqual(get_filtered.call_args.kwargs['backend'], 'threading')

    def test_invalid_backend_before_io(self):
        print('Testing invalid workflow backends are rejected before I/O')
        from squirrel.workflows.volume import filter_2d_workflow

        for backend, n_workers, error in ((None, 1, TypeError),
                                          ('invalid', 1, ValueError),
                                          ('serial', 2, ValueError)):
            with self.subTest(backend=backend, n_workers=n_workers):
                with patch('squirrel.library.io.load_data_handle') as load_handle:
                    with self.assertRaises(error):
                        filter_2d_workflow('input', 'output', filters=[['same', {}]],
                                           backend=backend, n_workers=n_workers)
                    load_handle.assert_not_called()

    def test_threading_and_multiprocessing_match_serial(self):
        print('Testing workflow output matches across execution backends')
        from squirrel.workflows.volume import filter_2d_workflow

        stack = np.arange(96, dtype=np.uint16).reshape(4, 4, 6)
        outputs = {}
        for backend, n_workers in (('serial', 1), ('threading', 2), ('multiprocessing', 2)):
            with self.subTest(backend=backend):
                with patch('squirrel.library.io.load_data_handle', return_value=(stack, stack.shape)), \
                     patch('squirrel.library.io.write_stack') as write_stack:
                    filter_2d_workflow('input', 'output', filters=[['same', {}]],
                                       backend=backend, n_workers=n_workers, progress=False)
                outputs[backend] = write_stack.call_args.args[1]
        for backend in ('threading', 'multiprocessing'):
            np.testing.assert_array_equal(outputs[backend], outputs['serial'])
