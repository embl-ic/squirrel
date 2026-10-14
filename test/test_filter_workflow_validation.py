import unittest
from unittest.mock import patch


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


if __name__ == '__main__':
    unittest.main()
