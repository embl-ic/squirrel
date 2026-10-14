import sys
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np


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


if __name__ == '__main__':
    unittest.main()
