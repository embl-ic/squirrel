import unittest
import warnings

import numpy as np


class TestNormalizationStack(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)
        self.stack = np.arange(3 * 8 * 8, dtype=np.uint8).reshape(3, 8, 8)

    def test_normalize_slices_backend_equivalence(self):
        print('Testing normalize_slices: serial and multiprocessing equivalence')
        from squirrel.library.normalization import normalize_slices

        serial = normalize_slices(self.stack, n_workers=1)
        parallel = normalize_slices(self.stack, n_workers=2)
        np.testing.assert_array_equal(serial, parallel)

    def test_adjust_greyscale_backend_equivalence(self):
        print('Testing adjust_greyscale: serial and multiprocessing equivalence')
        from squirrel.library.normalization import adjust_greyscale

        serial = adjust_greyscale(self.stack, n_workers=1, z_range=(1, 3))
        parallel = adjust_greyscale(self.stack, n_workers=2, z_range=(1, 3))
        np.testing.assert_array_equal(serial, parallel)
        self.assertEqual(serial.shape, (2, 8, 8))

    def test_clahe_backend_equivalence(self):
        print('Testing clahe_on_slices: serial and multiprocessing equivalence')
        from squirrel.library.normalization import clahe_on_slices

        serial = clahe_on_slices(self.stack, tile_grid_size=(2, 2), n_workers=1)
        parallel = clahe_on_slices(self.stack, tile_grid_size=(2, 2), n_workers=2)
        np.testing.assert_array_equal(serial, parallel)

    def test_clahe_multiprocessing_after_serial_calls(self):
        print('Testing clahe_on_slices: spawned workers after OpenCV initialization')
        import cv2
        from squirrel.library.normalization import clahe_on_slices

        cv2.getBuildInformation()
        serial = clahe_on_slices(self.stack, tile_grid_size=(2, 2), n_workers=1)
        parallel = clahe_on_slices(self.stack, tile_grid_size=(2, 2), n_workers=2)
        np.testing.assert_array_equal(serial, parallel)

    def test_invalid_worker_count(self):
        print('Testing stack processing: invalid worker count')
        from squirrel.library.normalization import adjust_greyscale

        with self.assertRaises(ValueError):
            adjust_greyscale(self.stack, n_workers=0)
