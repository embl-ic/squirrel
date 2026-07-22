import unittest
import warnings
import tempfile

import numpy as np
import os

from squirrel.library.io import load_data_handle, write_tif_stack
from squirrel.workflows.volume import crop_from_stack_workflow


class TestCropFromStackWorkflow(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_crop_from_stack_workflow_tiff(self):

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = tmp

            input_dir = os.path.join(tmp_path, "input")
            output_dir = os.path.join(tmp_path, "output")

            os.mkdir(input_dir)

            data = np.arange(
                5 * 10 * 10,
                dtype=np.uint16
            ).reshape(5, 10, 10)

            write_tif_stack(data, input_dir)

            crop_from_stack_workflow(
                stack_path=input_dir,
                out_path=output_dir,
                roi=[1, 3, 1, 4, 5, 3]
            )

            result, _ = load_data_handle(output_dir)

            np.testing.assert_array_equal(
                result[:],
                data[1:5, 3:8, 1:4]
            )

    def test_crop_from_stack_workflow_ome_zarr(self):

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = tmp

            input_dir = os.path.join(tmp_path, "input")
            output_dir = os.path.join(tmp_path, "output.ome.zarr")

            os.mkdir(input_dir)

            data = np.arange(
                5 * 10 * 10,
                dtype=np.uint16
            ).reshape(5, 10, 10)

            write_tif_stack(data, input_dir)

            crop_from_stack_workflow(
                stack_path=input_dir,
                out_path=output_dir,
                roi=[1, 3, 1, 4, 5, 3]
            )

            result, _ = load_data_handle(output_dir)

            np.testing.assert_array_equal(
                result[:],
                data[1:5, 3:8, 1:4]
            )