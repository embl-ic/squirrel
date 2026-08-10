import unittest
import warnings
from matplotlib.pylab import half
import numpy as np


def _create_random_segmentation(
        size=64,
        n_objects=200,
        warp_sigma=8,
        warp_amplitude=4,
        seed=42,
):
    
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)

    shape = (size, size, size)

    # ------------------------------------------------------------------
    # Random seed points
    # ------------------------------------------------------------------

    centers = rng.uniform(
        low=0,
        high=size,
        size=(n_objects, 3)
    )

    tree = cKDTree(centers)

    # ------------------------------------------------------------------
    # Smooth displacement field
    # ------------------------------------------------------------------

    dz = gaussian_filter(rng.standard_normal(shape), warp_sigma)
    dy = gaussian_filter(rng.standard_normal(shape), warp_sigma)
    dx = gaussian_filter(rng.standard_normal(shape), warp_sigma)

    for d in (dz, dy, dx):
        d *= warp_amplitude / np.std(d)

    # ------------------------------------------------------------------
    # Warp coordinates
    # ------------------------------------------------------------------

    z, y, x = np.indices(shape)

    coords = np.column_stack((
        (z + dz).ravel(),
        (y + dy).ravel(),
        (x + dx).ravel()
    ))

    # ------------------------------------------------------------------
    # Voronoi assignment
    # ------------------------------------------------------------------

    _, labels = tree.query(coords)

    seg = labels.reshape(shape).astype(np.uint32) + 1

    return seg


class TestLocalGraph(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_empty_tiles(self):

        print(f'Testing local graph of empty tiles ...')

        from squirrel.library.stitching import build_local_graph

        tile1 = np.zeros((512, 512, 512), dtype=np.uint32)
        tile2 = np.zeros((512, 512, 512), dtype=np.uint32)
        tile3 = np.zeros((512, 512, 512), dtype=np.uint32)
        tile4 = np.zeros((512, 512, 512), dtype=np.uint32)

        edges1, disaffinities1 = build_local_graph(
            seg=tile1,
            overlap=(1, 1, 1),
            right=tile2,
            bottom=tile3,
            behind=tile4,
            background=0,
            default_disaffinity=0.9
        )

        np.testing.assert_equal(edges1, np.empty((0, 2), dtype=np.uint64))
        np.testing.assert_equal(disaffinities1, np.array([], dtype=np.float32))

    def test_single_objects(self):

        print(f'Testing local graph of tiles with one object each ...')

        from squirrel.library.stitching import build_local_graph

        tile1 = np.ones((512, 512, 512), dtype=np.uint32)
        tile2 = np.ones((512, 512, 512), dtype=np.uint32) + 1
        tile3 = np.ones((512, 512, 512), dtype=np.uint32) + 2
        tile4 = np.ones((512, 512, 512), dtype=np.uint32) + 3

        edges1, disaffinities1 = build_local_graph(
            seg=tile1,
            overlap=(1, 1, 1),
            right=tile2,
            bottom=tile3,
            behind=tile4,
            background=0,
            default_disaffinity=0.9
        )

        expected_edges = np.array([[1, 2], [1, 3], [1, 4]], dtype=np.uint64)
        expected_disaffinities = np.array([0., 0., 0.], dtype=np.float32)

        np.testing.assert_equal(edges1, expected_edges)
        np.testing.assert_equal(disaffinities1, expected_disaffinities)

    def test_segmentation(self):

        print(f'Testing local graph of tiles with segmentation ...')

        from squirrel.library.stitching import build_local_graph

        seg = _create_random_segmentation(
            size=32,
            n_objects=6,
            seed=42
        )

        half = seg.shape[0] // 2

        # Create a simple segmentation in tile1
        tile1 = seg[:half, :half, :half]  # top left
        tile2 = seg[:half, :half, half:] + 10  # top right
        tile3 = seg[:half, half:, :half] + 20  # bottom left
        # tile4 = seg[:64, 64:, 64:]  # bottom right

        edges, disaffinities = build_local_graph(
            seg=tile1,
            overlap=(1, 1, 1),
            right=tile2,
            bottom=tile3,
            behind=None,
            background=0,
            default_disaffinity=0.9
        )

        edge_map = {
            tuple(edge): dis
            for edge, dis in zip(edges, disaffinities)
        }

        self.assertEqual(edges.shape[1], 2)
        self.assertEqual(len(edges), len(disaffinities))
        self.assertTrue(np.all(edges[:, 0] < edges[:, 1]))
        self.assertTrue(np.all(disaffinities >= 0))
        self.assertTrue(np.all(disaffinities <= 1))
        self.assertTrue(np.any(edges[:, 1] >= 10))
        self.assertTrue(np.any(edges[:, 1] >= 20))

        self.assertAlmostEqual(edge_map[(1, 11)], 0.142857, places=5)
        self.assertAlmostEqual(edge_map[(4, 24)], 0.75, places=5)


class TestGlobalMulticut(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_empty_tiles(self):

        print(f'Testing global multicut of empty tiles ...')

        edges = [np.empty((0, 2), dtype=np.uint64)] * 2
        disaffinities = [np.array([], dtype=np.float32)] * 2

        from squirrel.library.stitching import solve_global_multicut

        label_mapping = solve_global_multicut(
            edge_list=edges, 
            disaffinity_list=disaffinities, 
            beta=0.5
        )

        self.assertEqual(label_mapping, {})

    def test_single_objects(self):

        print(f'Testing global multicut of tiles with one object each ...')

        edges = [
            np.array([[1, 2], [1, 3], [1, 4]], dtype=np.uint64)
        ]
        disaffinities = [
            np.array([0., 0., 0.], dtype=np.float32)
        ]

        from squirrel.library.stitching import solve_global_multicut

        label_mapping = solve_global_multicut(
            edge_list=edges, 
            disaffinity_list=disaffinities, 
            beta=0.5
        )

        expected_mapping = {1: 1, 2: 1, 3: 1, 4: 1}

        self.assertEqual(label_mapping, expected_mapping)   

    def test_small_graphs(self):

        print(f'Testing small graphs ...')

        from squirrel.library.stitching import solve_global_multicut

        #
        # Case 1:
        #
        # 1 --0.0-- 2 --0.0-- 3
        #
        # All attractive.
        #

        edges = [np.array([
            [1, 2],
            [2, 3],
        ], dtype=np.uint64)]

        dis = [np.array([0.0, 0.0], dtype=np.float32)]

        mapping = solve_global_multicut(edges, dis, beta=0.5)

        self.assertEqual(mapping[1], mapping[2])
        self.assertEqual(mapping[2], mapping[3])

        #
        # Case 2:
        #
        # 1 --0.9-- 2 --0.9-- 3
        #
        # All repulsive.
        #

        dis = [np.array([0.9, 0.9], dtype=np.float32)]

        mapping = solve_global_multicut(edges, dis, beta=0.5)

        self.assertNotEqual(mapping[1], mapping[2])
        self.assertNotEqual(mapping[2], mapping[3])
        self.assertNotEqual(mapping[1], mapping[3])

        #
        # Case 3:
        #
        # 1 --0.0-- 2 --0.9-- 3
        #

        dis = [np.array([0.0, 0.9], dtype=np.float32)]

        mapping = solve_global_multicut(edges, dis, beta=0.5)

        self.assertEqual(mapping[1], mapping[2])
        self.assertNotEqual(mapping[2], mapping[3])

        #
        # Case 4:
        #
        #         2
        #       /   \
        #     0.0   0.0
        #     /       \
        #   1  --0.9--  3
        #

        edges = [np.array([
            [1, 2],
            [2, 3],
            [1, 3],
        ], dtype=np.uint64)]

        dis = [np.array([
            0.0,
            0.0,
            0.9,
        ], dtype=np.float32)]

        mapping = solve_global_multicut(edges, dis, beta=0.5)

        self.assertEqual(mapping[1], mapping[2])
        self.assertEqual(mapping[2], mapping[3])

        #
        # Case 5:
        #
        # Square with weak diagonal
        #
        # 1 --0.0-- 2
        # |  \      |
        # 0.0 0.9 0.0
        # |      \  |
        # 4 --0.0-- 3
        #

        edges = [np.array([
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
            [1, 3],
        ], dtype=np.uint64)]

        dis = [np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.9,
        ], dtype=np.float32)]

        mapping = solve_global_multicut(edges, dis, beta=0.5)
        
        self.assertEqual(len(set(mapping.values())), 1)


class TestRelabeling(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_relabel_segmentation(self):

        print(f'Testing relabeling of tiles ...')

        from squirrel.library.stitching import relabel_segmentation

        seg = _create_random_segmentation(
            size=32,
            n_objects=6,
            seed=42
        )

        label_mapping = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}

        relabeled = relabel_segmentation(
            seg,
            label_mapping,
            background=0
        )

        np.testing.assert_equal(relabeled, seg + 1)
