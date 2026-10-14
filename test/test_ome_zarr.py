import tempfile
import multiprocessing
import unittest
import warnings
from pathlib import Path

import numpy as np

from squirrel.library.ome_zarr import OMEZarrStore


def _write_ome_zarr_batch_worker(path, start, stop, result_queue):
    """Open the store independently, as a separate Snakemake worker would."""
    try:
        store = OMEZarrStore(path, mode="a")
        z, y, x = np.indices((stop - start, 16, 16))
        data = ((z + start) * 37 + y * 7 + x * 3).astype(np.uint16)
        store.write(0, (start, 0, 0), data, update_pyramid=True,
                    check_pyramid_alignment=True)
        result_queue.put((start, None))
    except Exception as exc:
        result_queue.put((start, repr(exc)))


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

    def test_iter_storage_blocks(self):

        blocks = list(
            self.store.iter_storage_blocks(0)
        )

        self.assertEqual(
            len(blocks),
            64,
        )

        self.assertEqual(
            tuple(blocks[0][0]),
            (0, 0, 0),
        )

        self.assertEqual(
            tuple(blocks[-1][0]),
            (48, 48, 48),
        )
    def test_parallel_rebuild(self):

        self.store.dataset(0)[:] = 3

        self.store.rebuild_pyramid(
            n_threads=4,
        )

        self.assertTrue(
            np.all(
                self.store.dataset(1)[:] == 3
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


    # -------------------------------------------------------------------------
    # AMST2-style batched writes and pyramid consistency
    # -------------------------------------------------------------------------

    def _make_batch_store(self, name):
        return OMEZarrStore.create(
            Path(self.tmpdir.name) / name,
            shape=(19, 16, 16),
            chunks=(1, 8, 8),
            shards=None,
            downsample_factors=((1, 2, 2), (1, 2, 2)),
            ome_version="0.4",
            zarr_format=2,
        )

    @staticmethod
    def _batch_data(start, stop):
        z, y, x = np.indices((stop - start, 16, 16))
        return ((z + start) * 37 + y * 7 + x * 3).astype(np.uint16)

    def test_batched_writes_match_sequential_and_rebuild(self):
        print('Testing OME-Zarr: aligned batch writes match sequential pyramid rebuild ...')
        batched = self._make_batch_store('batched.ome.zarr')
        reference = self._make_batch_store('reference.ome.zarr')
        for start, stop in ((0, 8), (8, 16), (16, 19)):
            batched.write(0, (start, 0, 0), self._batch_data(start, stop),
                          update_pyramid=True, check_pyramid_alignment=True)
        reference.dataset(0)[:] = self._batch_data(0, 19)
        reference.rebuild_pyramid()
        for level in range(3):
            np.testing.assert_array_equal(batched.dataset(level)[:], reference.dataset(level)[:])

    def test_unaligned_batch_is_rejected_before_writing(self):
        print('Testing OME-Zarr: pyramid alignment rejects unsafe batch boundaries ...')
        store = OMEZarrStore.create(
            Path(self.tmpdir.name) / 'unaligned.ome.zarr',
            shape=(16, 16, 16), chunks=(2, 8, 8), shards=None,
            downsample_factors=((2, 2, 2), (2, 2, 2)),
            ome_version='0.4', zarr_format=2,
        )
        data = self._batch_data(0, 6)
        with self.assertRaises(ValueError):
            store.write(0, (0, 0, 0), data, update_pyramid=True,
                        check_pyramid_alignment=True)
        np.testing.assert_array_equal(store.dataset(0)[:], 0)

    def test_concurrent_batch_writes_match_rebuilt_pyramid(self):
        print('Testing OME-Zarr: independent concurrent writers match a rebuilt pyramid ...')
        ctx = multiprocessing.get_context('spawn')
        batches = ((0, 8), (8, 16), (16, 19))
        for repetition in range(3):
            store = self._make_batch_store(f'parallel_{repetition}.ome.zarr')
            reference = self._make_batch_store(f'parallel_ref_{repetition}.ome.zarr')
            queue = ctx.Queue()
            workers = [ctx.Process(target=_write_ome_zarr_batch_worker,
                                   args=(str(Path(self.tmpdir.name) / f'parallel_{repetition}.ome.zarr'),
                                         start, stop, queue))
                       for start, stop in batches]
            try:
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(timeout=60)
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join()
                        self.fail('Concurrent OME-Zarr worker timed out')
                    self.assertEqual(worker.exitcode, 0)
                results = [queue.get(timeout=5) for _ in workers]
                self.assertTrue(all(error is None for _, error in results), results)
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join()
                queue.close()
                queue.join_thread()

            reference.dataset(0)[:] = self._batch_data(0, 19)
            reference.rebuild_pyramid()
            for level in range(3):
                np.testing.assert_array_equal(store.dataset(level)[:],
                                              reference.dataset(level)[:])

    def test_concurrent_retry_of_one_batch_matches_reference(self):
        print('Testing OME-Zarr: retry after concurrent batch writes preserves all pyramid levels ...')
        store = self._make_batch_store('parallel_retry.ome.zarr')
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()
        batches = ((0, 8), (8, 16), (16, 19))
        workers = [ctx.Process(target=_write_ome_zarr_batch_worker,
                               args=(str(Path(self.tmpdir.name) / 'parallel_retry.ome.zarr'),
                                     start, stop, queue)) for start, stop in batches]
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=60)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
                    self.fail('Concurrent OME-Zarr worker timed out')
                self.assertEqual(worker.exitcode, 0)
            results = [queue.get(timeout=5) for _ in workers]
            self.assertTrue(all(error is None for _, error in results), results)
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
            queue.close()
            queue.join_thread()
        store.write(0, (8, 0, 0), self._batch_data(8, 16),
                    update_pyramid=True, check_pyramid_alignment=True)
        reference = self._make_batch_store('parallel_retry_reference.ome.zarr')
        reference.dataset(0)[:] = self._batch_data(0, 19)
        reference.rebuild_pyramid()
        for level in range(3):
            np.testing.assert_array_equal(store.dataset(level)[:], reference.dataset(level)[:])

    def test_batch_retry_reproduces_pyramid(self):
        print('Testing OME-Zarr: overwriting a completed batch preserves pyramid consistency ...')
        store = self._make_batch_store('retry.ome.zarr')
        reference = self._make_batch_store('retry_reference.ome.zarr')
        for start, stop in ((0, 8), (8, 16), (16, 19)):
            store.write(0, (start, 0, 0), self._batch_data(start, stop),
                        update_pyramid=True, check_pyramid_alignment=True)
        store.write(0, (8, 0, 0), self._batch_data(8, 16),
                    update_pyramid=True, check_pyramid_alignment=True)
        reference.dataset(0)[:] = self._batch_data(0, 19)
        reference.rebuild_pyramid()
        for level in range(3):
            np.testing.assert_array_equal(store.dataset(level)[:], reference.dataset(level)[:])





# import tempfile
# import multiprocessing
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