import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from squirrel.library.ome_zarr import OMEZarrStore


class TestOMEZarr(unittest.TestCase):

    def setUp(self):

        warnings.simplefilter("ignore", category=Warning)

        self.tmpdir = tempfile.TemporaryDirectory()

        self.filename = Path(self.tmpdir.name) / "test.ome.zarr"

        self.store = OMEZarrStore.create(
            self.filename,
            shape=(64, 64, 64),
            chunks=(16, 16, 16),
            downsample_factors=(
                (2, 2, 2),
                (2, 2, 2),
            ),
            ome_version="0.4",
            zarr_format=2,
        )

    def tearDown(self):

        self.tmpdir.cleanup()

    # -------------------------------------------------------------------------
    # Creation
    # -------------------------------------------------------------------------

    def test_create_v2(self):

        self.assertEqual(
            self.store.metadata.ome_version,
            "0.4",
        )

        self.assertEqual(
            self.store.metadata.zarr_format,
            2,
        )

        self.assertEqual(
            self.store.shape(0),
            (64, 64, 64),
        )

        self.assertEqual(
            self.store.shape(1),
            (32, 32, 32),
        )

        self.assertEqual(
            self.store.shape(2),
            (16, 16, 16),
        )

    def test_create_v3(self):

        filename = (
            Path(self.tmpdir.name)
            / "test_v3.ome.zarr"
        )

        store = OMEZarrStore.create(
            filename,
            shape=(64, 64, 64),
            chunks=(16, 16, 16),
            downsample_factors=(
                (2, 2, 2),
                (2, 2, 2),
            ),
            ome_version="0.5",
            zarr_format=3,
        )

        self.assertEqual(
            store.metadata.ome_version,
            "0.5",
        )

        self.assertEqual(
            store.metadata.zarr_format,
            3,
        )

        self.assertEqual(
            store.shape(2),
            (16, 16, 16),
        )

    def test_create_v3_sharded(self):

        filename = (
            Path(self.tmpdir.name)
            / "test_v3_sharded.ome.zarr"
        )

        store = OMEZarrStore.create(
            filename,
            shape=(64, 64, 64),
            chunks=(16, 16, 16),
            shards=(32, 32, 32),
            downsample_factors=(
                (2, 2, 2),
                (2, 2, 2),
            ),
            ome_version="0.5",
            zarr_format=3,
        )

        self.assertEqual(
            store.metadata.ome_version,
            "0.5",
        )

        self.assertEqual(
            store.metadata.zarr_format,
            3,
        )

        self.assertEqual(
            tuple(store.shards(0)),
            (32, 32, 32),
        )

        self.assertEqual(
            tuple(store.chunks(0)),
            (16, 16, 16),
        )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    def test_downsample_factors(self):

        self.assertEqual(
            self.store.metadata.downsample_factors,
            [
                (2, 2, 2),
                (2, 2, 2),
            ],
        )

    def test_levels(self):

        self.assertEqual(
            self.store.metadata.levels,
            ["s0", "s1", "s2"],
        )

    def test_scales(self):

        self.assertEqual(
            self.store.metadata.scales,
            [
                [1., 1., 1.],
                [2., 2., 2.],
                [4., 4., 4.],
            ],
        )

    # -------------------------------------------------------------------------
    # Read / write
    # -------------------------------------------------------------------------

    def test_write_read_roi(self):

        data = np.random.randint(
            0,
            100,
            size=(13, 17, 9),
            dtype=np.uint16,
        )

        position = (7, 11, 5)

        self.store.write(
            0,
            position,
            data,
        )

        result = self.store.read(
            0,
            position,
            data.shape,
        )

        np.testing.assert_array_equal(
            result,
            data,
        )

    def test_write_update_pyramid(self):

        data = np.ones(
            (16, 16, 16),
            dtype=np.uint16,
        )

        self.store.write(
            0,
            (0, 0, 0),
            data,
            update_pyramid=True,
        )

        self.assertTrue(
            np.all(
                self.store.dataset(1)[:8, :8, :8] == 1
            )
        )

    def test_rebuild_pyramid(self):

        self.store.dataset(0)[:] = 5

        self.store.rebuild_pyramid()

        self.assertTrue(
            np.all(
                self.store.dataset(1)[:] == 5
            )
        )

    # -------------------------------------------------------------------------
    # Alignment
    # -------------------------------------------------------------------------

    def test_alignment_check(self):

        with self.assertRaises(ValueError):

            self.store.write(
                0,
                (0, 0, 0),
                np.ones(
                    (10, 16, 16),
                    dtype=np.uint16,
                ),
                check_alignment=True,
            )

    def test_check_pyramid_alignment_passes(self):

        self.store.write(
            0,
            (0, 0, 0),
            np.ones(
                (64, 64, 64),
                dtype=np.uint16,
            ),
            check_pyramid_alignment=True,
        )

    def test_check_pyramid_alignment_fails(self):

        with self.assertRaises(ValueError):

            self.store.write(
                0,
                (0, 0, 0),
                np.ones(
                    (32, 32, 32),
                    dtype=np.uint16,
                ),
                check_pyramid_alignment=True,
            )

    # -------------------------------------------------------------------------
    # Empty overwrite protection
    # -------------------------------------------------------------------------

    def test_require_empty(self):

        data = np.ones(
            (8, 8, 8),
            dtype=np.uint16,
        )

        self.store.write(
            0,
            (0, 0, 0),
            data,
        )

        with self.assertRaises(ValueError):

            self.store.write(
                0,
                (0, 0, 0),
                data,
                require_empty=True,
            )


# import tempfile
# import unittest
# import warnings
# from pathlib import Path

# import numpy as np

# from squirrel.library.ome_zarr_new import OMEZarrStore, expand_roi, check_grid_alignment


# class TestOMEZarr(unittest.TestCase):

#     def setUp(self):

#         warnings.simplefilter('ignore', category=Warning)

#         self.tmpdir = tempfile.TemporaryDirectory()

#         self.filename = Path(self.tmpdir.name) / "test.ome.zarr"

#         self.store = OMEZarrStore.create(
#             self.filename,
#             shape=(64, 64, 64),
#             chunks=(16, 16, 16),
#             downsample_factors=(2, 2),
#             resolution=(0.5, 0.2, 0.2),
#             unit="micrometer",
#             downsample_method="Sample",
#         )

#     def tearDown(self):

#         self.tmpdir.cleanup()

#     def test_create(self):

#         self.assertEqual(self.store.shape(0), (64, 64, 64))
#         self.assertEqual(self.store.shape(1), (32, 32, 32))
#         self.assertEqual(self.store.shape(2), (16, 16, 16))

#         self.assertEqual(
#             self.store.metadata.downsample_factors,
#             [(2, 2, 2), (2, 2, 2)],
#         )

#     def test_write_read_roi(self):

#         data = np.random.randint(
#             0,
#             100,
#             size=(13, 17, 9),
#             dtype=np.uint16,
#         )

#         position = (7, 11, 5)

#         self.store.write(
#             0,
#             position,
#             data,
#         )

#         result = self.store.read(
#             0,
#             position,
#             data.shape,
#         )

#         np.testing.assert_array_equal(
#             result,
#             data,
#         )

#     def test_grid_alignment_check(self):

#         data = np.ones(
#             (10, 16, 16),
#             dtype=np.uint16,
#         )

#         with self.assertRaises(ValueError):

#             self.store.write(
#                 0,
#                 (0, 0, 0),
#                 data,
#                 check_alignment=True,
#             )

#     def test_require_empty(self):

#         data = np.ones(
#             (8, 8, 8),
#             dtype=np.uint16,
#         )

#         self.store.write(
#             0,
#             (0, 0, 0),
#             data,
#         )

#         with self.assertRaises(ValueError):

#             self.store.write(
#                 0,
#                 (0, 0, 0),
#                 data,
#                 require_empty=True,
#             )
    
#     def test_update_pyramid(self):

#         data = np.ones(
#             (16, 16, 16),
#             dtype=np.uint16,
#         ) * 42

#         self.store.write(
#             0,
#             (0, 0, 0),
#             data,
#             update_pyramid=True,
#         )

#         result1 = self.store.read(
#             1,
#             (0, 0, 0),
#             (8, 8, 8),
#         )

#         result2 = self.store.read(
#             2,
#             (0, 0, 0),
#             (4, 4, 4),
#         )

#         np.testing.assert_array_equal(
#             result1,
#             np.ones((8, 8, 8), dtype=np.uint16) * 42,
#         )

#         np.testing.assert_array_equal(
#             result2,
#             np.ones((4, 4, 4), dtype=np.uint16) * 42,
#         )

#     def test_metadata(self):

#         self.assertEqual(
#             self.store.metadata.axes[0]["unit"],
#             "micrometer",
#         )

#         self.assertEqual(
#             self.store.metadata.axes[1]["unit"],
#             "micrometer",
#         )

#         self.assertEqual(
#             self.store.metadata.downsample_method,
#             "Sample",
#         )

#         self.assertEqual(
#             self.store.metadata._scales[0],
#             [0.5, 0.2, 0.2],
#         )

#     def test_check_pyramid_alignment(self):

#         self.store.check_pyramid_alignment(
#             0,
#             (0, 0, 0),
#             (64, 64, 64),
#         )

#         with self.assertRaises(ValueError):

#             self.store.check_pyramid_alignment(
#                 0,
#                 (0, 0, 0),
#                 (16, 16, 16),
#             )

#     def test_update_pyramid_arbitrary_roi(self):

#         data = np.random.randint(
#             0,
#             100,
#             size=(13, 17, 9),
#             dtype=np.uint16,
#         )

#         self.store.write(
#             0,
#             (7, 11, 5),
#             data,
#             update_pyramid=True,
#         )

#     def test_expand_roi(self):

#         start, shape = expand_roi(
#             (5, 7, 9),
#             (13, 17, 9),
#             (2, 2, 2),
#         )

#         np.testing.assert_array_equal(
#             start,
#             (4, 6, 8),
#         )

#         np.testing.assert_array_equal(
#             shape,
#             (14, 18, 10),
#         )

#     def test_storage_grid_v2(self):

#         np.testing.assert_array_equal(
#             self.store.metadata.storage_grid(
#                 self.store.dataset(0)
#             ),
#             (16, 16, 16),
#         )

#         np.testing.assert_array_equal(
#             self.store.metadata.storage_grid(
#                 self.store.dataset(1)
#             ),
#             (16, 16, 16),
#         )

#     def test_grid_alignment(self):

#         self.assertTrue(
#             check_grid_alignment(
#                 (0, 0, 0),
#                 (16, 16, 16),
#                 (16, 16, 16),
#                 (64, 64, 64),
#             )
#         )

#         self.assertFalse(
#             check_grid_alignment(
#                 (1, 0, 0),
#                 (16, 16, 16),
#                 (16, 16, 16),
#                 (64, 64, 64),
#             )
#         )  