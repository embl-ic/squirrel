import unittest
import warnings
import tempfile
from pathlib import Path
import numpy as np

from squirrel.library.serial_em import (
    get_unique_key,
    get_value_from_item,
    get_value_list_from_item,
    match_regex,
    navigator_file_to_dict,
    extend_navigator_dict,
    get_filepath_from_nav_item,
    get_map_filepath_from_nav_item,
    get_map_items_by_glob,
    get_map_scale_xy,
    get_map_scale_matrix_from_item,
    get_map_shape_from_serialem_item,
    get_mrc_shape,
    get_resolution_from_mrc_header,
    get_contrast_limits_from_map,
    NavigatorMap,
    MapCollection,
    MatchRule,
    MapHierarchy,
    SingleParticleProfile,
    NavigatorProfile,
    TomoCLEMProfile,
    Navigator
)


class TestSerialEMUtilities(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_get_unique_key(self):
        print('Testing SerialEM utilities: get_unique_key ...')

        items = {
            'item': {},
            'item-1': {},
            'item-2': {},
        }

        self.assertEqual(get_unique_key('new_item', items), 'new_item')
        self.assertEqual(get_unique_key('item', items), 'item-3')

    def test_get_value_from_item(self):
        print('Testing SerialEM utilities: get_value_from_item ...')

        item = {
            'MapBinning': '4',
            'Exposure': '1.25',
        }

        self.assertEqual(get_value_from_item(item, 'MapBinning'), 4.0)
        self.assertEqual(get_value_from_item(item, 'Exposure'), 1.25)
        self.assertIsNone(get_value_from_item(item, 'Missing'))

    def test_get_value_list_from_item(self):
        print('Testing SerialEM utilities: get_value_list_from_item ...')

        item = {
            'StageXYZ': '1.5 -2.0 3.25',
            'MultipleSpaces': '1.0   2.0    3.0',
            'Tabs': '1.0\t2.0\t3.0',
        }

        self.assertEqual(
            get_value_list_from_item(item, 'StageXYZ'),
            [1.5, -2.0, 3.25],
        )

        self.assertEqual(
            get_value_list_from_item(item, 'MultipleSpaces'),
            [1.0, 2.0, 3.0],
        )

        self.assertEqual(
            get_value_list_from_item(item, 'Tabs'),
            [1.0, 2.0, 3.0],
        )

        self.assertIsNone(
            get_value_list_from_item(item, 'Missing')
        )

    def test_match_regex(self):
        print('Testing SerialEM utilities: match_regex ...')

        self.assertEqual(
            match_regex(
                'L03_tgt_012_view.mrc',
                r'L(\d{2})_tgt_(\d{3})',
            ),
            '03012',
        )

        self.assertEqual(
            match_regex(
                'Grid 01 some_name',
                r'Grid \d{2} (\S+)',
            ),
            'some_name',
        )

        self.assertEqual(
            match_regex(
                'L03_tgt_012_view.mrc',
                r'L\d{2}',
            ),
            'L03',
        )

        self.assertIsNone(
            match_regex(
                'L03_tgt_012_view.mrc',
                r'not_present',
            )
        )

    def test_navigator_file_to_dict(self):
        print('Testing SerialEM utilities: navigator_file_to_dict ...')

        content = """
AdocVersion = 2.00
LastSavedAs = test.nav

[Item = 1]
MapID = 101
MapFile = grid.mrc
StageXYZ = 1.0 2.0 3.0

[Item = 2]
MapID = 102
MapFile = view.mrc
StageXYZ = 4.0 5.0 6.0
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.nav'
            filepath.write_text(content)

            nav_dict = navigator_file_to_dict(filepath)

        self.assertEqual(nav_dict['AdocVersion'], 2.0)
        self.assertEqual(nav_dict['LastSavedAs'], 'test.nav')

        self.assertEqual(len(nav_dict['items']), 2)

        self.assertEqual(
            nav_dict['items']['1']['MapID'],
            101,
        )

        self.assertEqual(
            nav_dict['items']['1']['MapFile'],
            'grid.mrc',
        )

        self.assertEqual(
            nav_dict['items']['1']['StageXYZ'],
            '1.0 2.0 3.0',
        )

        self.assertEqual(
            nav_dict['items']['2']['MapID'],
            102,
        )

    def test_navigator_file_to_dict_duplicate_item_names(self):
        print(
            'Testing SerialEM utilities: '
            'navigator_file_to_dict duplicate item names ...'
        )

        content = """
AdocVersion = 2.00

[Item = 1]
MapID = 101

[Item = 1]
MapID = 102

[Item = 1]
MapID = 103
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.nav'
            filepath.write_text(content)

            nav_dict = navigator_file_to_dict(filepath)

        self.assertEqual(
            list(nav_dict['items'].keys()),
            ['1', '1-1', '1-2'],
        )

        self.assertEqual(
            nav_dict['items']['1']['MapID'],
            101,
        )

        self.assertEqual(
            nav_dict['items']['1-1']['MapID'],
            102,
        )

        self.assertEqual(
            nav_dict['items']['1-2']['MapID'],
            103,
        )

    def test_extend_navigator_dict_new_item(self):
        print(
            'Testing SerialEM utilities: '
            'extend_navigator_dict new item ...'
        )

        nav_dict = {
            'AdocVersion': 2.0,
            'items': {
                '1': {
                    'MapID': 101,
                    'MapFile': 'grid.mrc',
                },
            },
        }

        content = """
AdocVersion = 2.00

[Item = 2]
MapID = 102
MapFile = view.mrc
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'extend.nav'
            filepath.write_text(content)

            result = extend_navigator_dict(
                filepath,
                nav_dict,
            )

        self.assertEqual(
            list(result['items'].keys()),
            ['1', '2'],
        )

        self.assertEqual(
            result['items']['2']['MapID'],
            102,
        )

    def test_extend_navigator_dict_duplicate_item(self):
        print(
            'Testing SerialEM utilities: '
            'extend_navigator_dict duplicate item ...'
        )

        nav_dict = {
            'AdocVersion': 2.0,
            'items': {
                '1': {
                    'MapID': 101,
                    'MapFile': 'grid.mrc',
                },
            },
        }

        content = """
AdocVersion = 2.00

[Item = 1]
MapID = 101
MapFile = grid.mrc
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'extend.nav'
            filepath.write_text(content)

            result = extend_navigator_dict(
                filepath,
                nav_dict,
            )

        self.assertEqual(len(result['items']), 1)
        self.assertEqual(
            result['items']['1']['MapID'],
            101,
        )

    def test_extend_navigator_dict_same_key_different_map_id(self):
        print(
            'Testing SerialEM utilities: '
            'extend_navigator_dict same key different MapID ...'
        )

        nav_dict = {
            'AdocVersion': 2.0,
            'items': {
                '1': {
                    'MapID': 101,
                    'MapFile': 'grid.mrc',
                },
            },
        }

        content = """
AdocVersion = 2.00

[Item = 1]
MapID = 102
MapFile = view.mrc
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'extend.nav'
            filepath.write_text(content)

            result = extend_navigator_dict(
                filepath,
                nav_dict,
            )

        self.assertEqual(
            list(result['items'].keys()),
            ['1', '1-1'],
        )

        self.assertEqual(
            result['items']['1']['MapID'],
            101,
        )

        self.assertEqual(
            result['items']['1-1']['MapID'],
            102,
        )

    def test_extend_navigator_dict_different_adoc_version(self):
        print(
            'Testing SerialEM utilities: '
            'extend_navigator_dict different AdocVersion ...'
        )

        nav_dict = {
            'AdocVersion': 2.0,
            'items': {},
        }

        content = """
AdocVersion = 3.00
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'extend.nav'
            filepath.write_text(content)

            with self.assertRaises(ValueError):
                extend_navigator_dict(
                    filepath,
                    nav_dict,
                )

    def test_extend_navigator_dict_conflicting_map_file(self):
        print(
            'Testing SerialEM utilities: '
            'extend_navigator_dict conflicting MapFile ...'
        )

        nav_dict = {
            'AdocVersion': 2.0,
            'items': {
                '1': {
                    'MapID': 101,
                    'MapFile': 'grid.mrc',
                },
            },
        }

        content = """
AdocVersion = 2.00

[Item = 1]
MapID = 101
MapFile = different_grid.mrc
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'extend.nav'
            filepath.write_text(content)

            with self.assertRaises(ValueError):
                extend_navigator_dict(
                    filepath,
                    nav_dict,
                )

    def test_get_filepath_from_nav_item(self):
        print(
            'Testing SerialEM utilities: '
            'get_filepath_from_nav_item ...'
        )

        nav_filepath = Path('/data/experiment/test.nav')

        item = {
            'SomeFile': r'Z:\acquisition\data\image.mrc',
        }

        result = get_filepath_from_nav_item(
            nav_filepath,
            item,
            'SomeFile',
        )

        self.assertEqual(
            result,
            Path('/data/experiment/image.mrc'),
        )

    def test_get_map_filepath_from_nav_item(self):
        print(
            'Testing SerialEM utilities: '
            'get_map_filepath_from_nav_item ...'
        )

        nav_filepath = Path('/data/experiment/test.nav')

        item = {
            'MapFile': r'Z:\acquisition\data\grid.mrc',
        }

        result = get_map_filepath_from_nav_item(
            nav_filepath,
            item,
        )

        self.assertEqual(
            result,
            Path('/data/experiment/grid.mrc'),
        )

    def test_get_map_items_by_glob(self):
        print(
            'Testing SerialEM utilities: '
            'get_map_items_by_glob ...'
        )

        nav_dict = {
            'items': {
                '1': {
                    'MapFile': r'Z:\data\gridmap.st',
                },
                '2': {
                    'MapFile': r'Z:\data\L01_search.mrc',
                },
                '3': {
                    'MapFile': r'Z:\data\L02_search.mrc',
                },
                '4': {
                    'MapFile': r'Z:\data\L01_view.mrc',
                },
                '5': {
                    'Note': 'Not a map',
                },
            },
        }

        result = get_map_items_by_glob(
            nav_dict,
            Path('/data/experiment/test.nav'),
            '*_search.mrc',
        )

        self.assertEqual(
            list(result.keys()),
            ['2', '3'],
        )

    def test_get_map_items_by_glob_all_mrc(self):
        print(
            'Testing SerialEM utilities: '
            'get_map_items_by_glob all MRC files ...'
        )

        nav_dict = {
            'items': {
                '1': {
                    'MapFile': 'gridmap.st',
                },
                '2': {
                    'MapFile': 'search.mrc',
                },
                '3': {
                    'MapFile': 'view.mrc',
                },
            },
        }

        result = get_map_items_by_glob(
            nav_dict,
            Path('/data/experiment/test.nav'),
        )

        self.assertEqual(
            list(result.keys()),
            ['2', '3'],
        )
                
    def test_get_map_scale_xy(self):
        print(
            'Testing SerialEM utilities: '
            'get_map_scale_xy ...'
        )

        item = {
            'StageXYZ': '12.5 -4.25 1.75',
        }

        result = get_map_scale_xy(item)

        self.assertEqual(
            result,
            [12.5, -4.25],
        )

    def test_get_map_scale_xy_missing(self):
        print(
            'Testing SerialEM utilities: '
            'get_map_scale_xy missing ...'
        )

        result = get_map_scale_xy({})

        self.assertIsNone(result)

    def test_get_map_scale_matrix_from_item(self):
        print(
            'Testing SerialEM utilities: '
            'get_map_scale_matrix_from_item ...'
        )

        item = {
            'MapScaleMat': '1.0 2.0 3.0 4.0',
        }

        result = get_map_scale_matrix_from_item(item)

        expected = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])

        self.assertTrue(
            np.allclose(result, expected)
        )

    def test_get_map_scale_matrix_from_item_missing(self):
        print(
            'Testing SerialEM utilities: '
            'get_map_scale_matrix_from_item missing ...'
        )

        result = get_map_scale_matrix_from_item({})

        self.assertIsNone(result)

    def test_get_map_scale_matrix_from_item_invalid(self):
        print(
            'Testing SerialEM utilities: '
            'get_map_scale_matrix_from_item invalid ...'
        )

        item = {
            'MapScaleMat': '1.0 2.0 3.0',
        }

        with self.assertRaises(ValueError):
            get_map_scale_matrix_from_item(item)

    def test_get_map_shape_from_serialem_item(self):
        print(
            'Testing SerialEM utilities: '
            'get_map_shape_from_serialem_item ...'
        )

        item = {
            'MapBinning': '4',
            'MontBinning': '2',
            'MapWidthHeight': '1000 800',
        }

        result = get_map_shape_from_serialem_item(
            item,
            binning=2,
        )

        expected = np.array([
            1000,
            800,
        ])

        self.assertTrue(
            np.array_equal(result, expected)
        )

    def test_get_mrc_shape(self):
        print(
            'Testing SerialEM utilities: '
            'get_mrc_shape ...'
        )

        import mrcfile

        data = np.zeros(
            (3, 20, 30),
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.mrc'

            with mrcfile.new(filepath, overwrite=True) as mrc:
                mrc.set_data(data)

            result = get_mrc_shape(filepath)

        self.assertEqual(
            result,
            (3, 20, 30),
        )

    def test_get_resolution_from_mrc_header(self):
        print(
            'Testing SerialEM utilities: '
            'get_resolution_from_mrc_header ...'
        )

        import mrcfile

        data = np.zeros(
            (1, 20, 30),
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.mrc'

            with mrcfile.new(filepath, overwrite=True) as mrc:
                mrc.set_data(data)
                mrc.voxel_size = 10.0

            result = get_resolution_from_mrc_header(
                filepath,
                unit='micrometer',
            )

        self.assertAlmostEqual(
            result,
            0.001,
        )

    def test_get_resolution_from_mrc_header(self):
        print(
            'Testing SerialEM utilities: '
            'get_resolution_from_mrc_header ...'
        )

        import mrcfile

        data = np.zeros(
            (3, 20, 30),
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.mrc'

            with mrcfile.new(filepath, overwrite=True) as mrc:
                mrc.set_data(data)

                mrc.voxel_size = (
                    10.0,
                    20.0,
                    30.0,
                )

            result = get_resolution_from_mrc_header(
                filepath,
                unit='micrometer',
            )

        expected = np.array([
            0.001,
            0.002,
            0.003,
        ])

        self.assertTrue(
            np.allclose(result, expected)
        )

    def test_get_map_shape_from_serialem_item_binning(self):
        print(
            'Testing SerialEM utilities: '
            'get_map_shape_from_serialem_item binning ...'
        )

        item = {
            'MapBinning': '4',
            'MontBinning': '2',
            'MapWidthHeight': '1200 800',
        }

        result = get_map_shape_from_serialem_item(
            item,
            binning=1,
        )

        expected = np.array([
            2400,
            1600,
        ])

        self.assertTrue(
            np.array_equal(result, expected)
        )
        

class TestNavigatorMap(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter(
            'ignore',
            category=Warning,
        )

    def test_note(self):
        print('Testing NavigatorMap: note ...')

        map_ = NavigatorMap(
            id='1',
            map_type='view',
            serialem_item={
                'Note': 'Sec 10 test',
            },
            source_filepath=Path('source.mrc'),
            filepath=Path('map.mrc'),
        )

        self.assertEqual(
            map_.note,
            'Sec 10 test',
        )

    def test_note_missing(self):
        print('Testing NavigatorMap: note missing ...')

        map_ = NavigatorMap(
            id='1',
            map_type='view',
            serialem_item={},
            source_filepath=Path('source.mrc'),
            filepath=Path('map.mrc'),
        )

        self.assertIsNone(
            map_.note,
        )

    def test_stage_xy(self):
        print('Testing NavigatorMap: stage_xy ...')

        map_ = NavigatorMap(
            id='1',
            map_type='view',
            serialem_item={
                'StageXYZ': '12.5 -4.25 1.75',
            },
            source_filepath=Path('source.mrc'),
            filepath=Path('map.mrc'),
        )

        expected = np.array([
            12.5,
            -4.25,
        ])

        self.assertTrue(
            np.allclose(
                map_.stage_xy,
                expected,
            )
        )

    def test_stage_xy_missing(self):
        print('Testing NavigatorMap: stage_xy missing ...')

        map_ = NavigatorMap(
            id='1',
            map_type='view',
            serialem_item={},
            source_filepath=Path('source.mrc'),
            filepath=Path('map.mrc'),
        )

        self.assertIsNone(
            map_.stage_xy,
        )

    def test_map_scale_matrix(self):
        print('Testing NavigatorMap: map_scale_matrix ...')

        map_ = NavigatorMap(
            id='1',
            map_type='view',
            serialem_item={
                'MapScaleMat': '1.0 2.0 3.0 4.0',
            },
            source_filepath=Path('source.mrc'),
            filepath=Path('map.mrc'),
        )

        expected = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])

        self.assertTrue(
            np.allclose(
                map_.map_scale_matrix,
                expected,
            )
        )

    def test_has_serialem_key(self):
        print('Testing NavigatorMap: has_serialem_key ...')

        map_ = NavigatorMap(
            id='1',
            map_type='view',
            serialem_item={
                'MapID': 123,
            },
            source_filepath=Path('source.mrc'),
            filepath=Path('map.mrc'),
        )

        self.assertTrue(
            map_.has_serialem_key('MapID')
        )

        self.assertFalse(
            map_.has_serialem_key('Note')
        )

    def test_get_serialem_value(self):
        print('Testing NavigatorMap: get_serialem_value ...')

        map_ = NavigatorMap(
            id='1',
            map_type='view',
            serialem_item={
                'MapID': 123,
                'MapFile': 'test.mrc',
            },
            source_filepath=Path('source.mrc'),
            filepath=Path('map.mrc'),
        )

        self.assertEqual(
            map_.get_serialem_value('MapID'),
            123,
        )

        self.assertEqual(
            map_.get_serialem_value(
                'Missing',
                default='default',
            ),
            'default',
        )

    def test_load_shape(self):
        print('Testing NavigatorMap: load_shape ...')

        import mrcfile

        data = np.zeros(
            (3, 20, 30),
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.mrc'

            with mrcfile.new(
                filepath,
                overwrite=True,
            ) as mrc:
                mrc.set_data(data)

            map_ = NavigatorMap(
                id='1',
                map_type='view',
                serialem_item={},
                source_filepath=filepath,
                filepath=filepath,
            )

            result = map_.load_shape()

        self.assertEqual(
            result,
            (3, 20, 30),
        )

        self.assertEqual(
            map_.shape,
            (3, 20, 30),
        )

    def test_load_resolution(self):
        print('Testing NavigatorMap: load_resolution ...')

        import mrcfile

        data = np.zeros(
            (3, 20, 30),
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.mrc'

            with mrcfile.new(
                filepath,
                overwrite=True,
            ) as mrc:
                mrc.set_data(data)
                mrc.voxel_size = (
                    10.0,
                    20.0,
                    30.0,
                )

            map_ = NavigatorMap(
                id='1',
                map_type='view',
                serialem_item={},
                source_filepath=filepath,
                filepath=filepath,
            )

            result = map_.load_resolution()

        expected = np.array([
            0.001,
            0.002,
            0.003,
        ])

        self.assertTrue(
            np.allclose(
                result,
                expected,
            )
        )

        self.assertTrue(
            np.allclose(
                map_.resolution,
                expected,
            )
        )

    def test_load_resolution_force(self):
        print('Testing NavigatorMap: load_resolution force ...')

        import mrcfile

        data = np.zeros(
            (1, 10, 10),
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.mrc'

            with mrcfile.new(
                filepath,
                overwrite=True,
            ) as mrc:
                mrc.set_data(data)
                mrc.voxel_size = (
                    10.0,
                    10.0,
                    10.0,
                )

            map_ = NavigatorMap(
                id='1',
                map_type='view',
                serialem_item={},
                source_filepath=filepath,
                filepath=filepath,
            )

            resolution_1 = map_.load_resolution()

            with mrcfile.open(
                filepath,
                mode='r+',
            ) as mrc:
                mrc.voxel_size = (
                    20.0,
                    20.0,
                    20.0,
                )

            resolution_cached = map_.load_resolution()

            resolution_refreshed = map_.load_resolution(
                force=True,
            )

        self.assertTrue(
            np.allclose(
                resolution_1,
                [0.001, 0.001, 0.001],
            )
        )

        self.assertTrue(
            np.allclose(
                resolution_cached,
                [0.001, 0.001, 0.001],
            )
        )

        self.assertTrue(
            np.allclose(
                resolution_refreshed,
                [0.002, 0.002, 0.002],
            )
    )

    def test_refresh_metadata(self):
        print('Testing NavigatorMap: refresh_metadata ...')

        import mrcfile

        data = np.zeros(
            (3, 20, 30),
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.mrc'

            with mrcfile.new(
                filepath,
                overwrite=True,
            ) as mrc:
                mrc.set_data(data)
                mrc.voxel_size = (
                    10.0,
                    20.0,
                    30.0,
                )

            map_ = NavigatorMap(
                id='1',
                map_type='view',
                serialem_item={},
                source_filepath=filepath,
                filepath=filepath,
            )

            map_.resolution = np.array([
                1.0,
                1.0,
                1.0,
            ])
            map_.shape = (
                1,
                1,
                1,
            )
            map_.contrast_limits = (
                -1.0,
                -1.0,
            )

            map_.refresh_metadata()

            self.assertTrue(
                np.allclose(
                    map_.resolution,
                    [
                        0.001,
                        0.002,
                        0.003,
                    ],
                )
            )

            self.assertEqual(
                map_.shape,
                (
                    3,
                    20,
                    30,
                ),
            )

            # Contrast limits should not be refreshed by default.
            self.assertEqual(
                map_.contrast_limits,
                (
                    -1.0,
                    -1.0,
                ),
            )

    def test_refresh_metadata_with_contrast_limits(self):
        print(
            'Testing NavigatorMap: '
            'refresh_metadata with contrast limits ...'
        )

        import mrcfile

        data = np.arange(
            3 * 20 * 30,
            dtype=np.float32,
        ).reshape(
            3,
            20,
            30,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.mrc'

            with mrcfile.new(
                filepath,
                overwrite=True,
            ) as mrc:
                mrc.set_data(data)
                mrc.voxel_size = (
                    10.0,
                    20.0,
                    30.0,
                )

            map_ = NavigatorMap(
                id='1',
                map_type='view',
                serialem_item={},
                source_filepath=filepath,
                filepath=filepath,
            )

            map_.contrast_limits = (
                -1.0,
                -1.0,
            )

            map_.refresh_metadata(
                include_contrast_limits=True,
            )

            expected = tuple(
                float(value)
                for value in get_contrast_limits_from_map(filepath)
            )

            self.assertEqual(
                map_.contrast_limits,
                expected,
            )


class TestMapCollection(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter(
            'ignore',
            category=Warning,
        )

        self.grid = NavigatorMap(
            id='grid1',
            map_type='grid',
            serialem_item={
                'MapFile': 'grid.mrc',
                'Note': 'Grid 01',
            },
            source_filepath='grid_source.mrc',
            filepath='grid.mrc',
        )

        self.search1 = NavigatorMap(
            id='search1',
            map_type='search',
            serialem_item={
                'MapFile': 'L01_search.mrc',
                'Note': 'Sec 1',
            },
            source_filepath='search1_source.mrc',
            filepath='L01_search.mrc',
        )

        self.search2 = NavigatorMap(
            id='search2',
            map_type='search',
            serialem_item={
                'MapFile': 'L02_search.mrc',
                'Note': 'Sec 2',
            },
            source_filepath='search2_source.mrc',
            filepath='L02_search.mrc',
        )

    def test_add_and_get(self):
        print('Testing MapCollection: add and get ...')

        maps = MapCollection()

        maps.add(self.grid)

        result = maps.get(
            'grid',
            'grid1',
        )

        self.assertIs(
            result,
            self.grid,
        )

    def test_add_duplicate(self):
        print('Testing MapCollection: add duplicate ...')

        maps = MapCollection()

        maps.add(self.grid)

        with self.assertRaises(ValueError):
            maps.add(self.grid)

    def test_add_many(self):
        print('Testing MapCollection: add_many ...')

        maps = MapCollection()

        maps.add_many([
            self.grid,
            self.search1,
            self.search2,
        ])

        self.assertEqual(
            len(maps),
            3,
        )

    def test_remove(self):
        print('Testing MapCollection: remove ...')

        maps = MapCollection()

        maps.add(self.grid)

        result = maps.remove(
            'grid',
            'grid1',
        )

        self.assertIs(
            result,
            self.grid,
        )

        self.assertNotIn(
            ('grid', 'grid1'),
            maps,
        )

        self.assertNotIn(
            'grid',
            maps.map_types(),
        )

    def test_get_missing(self):
        print('Testing MapCollection: get missing ...')

        maps = MapCollection()

        with self.assertRaises(KeyError):
            maps.get(
                'view',
                'missing',
            )

    def test_get_optional(self):
        print('Testing MapCollection: get_optional ...')

        maps = MapCollection()
        maps.add(self.grid)

        self.assertIs(
            maps.get_optional(
                'grid',
                'grid1',
            ),
            self.grid,
        )

        self.assertIsNone(
            maps.get_optional(
                'grid',
                'missing',
            )
        )

    def test_by_type(self):
        print('Testing MapCollection: by_type ...')

        maps = MapCollection()

        maps.add_many([
            self.grid,
            self.search1,
            self.search2,
        ])

        result = maps.by_type('search')

        self.assertEqual(
            list(result.keys()),
            [
                'search1',
                'search2',
            ],
        )

    def test_ids(self):
        print('Testing MapCollection: ids ...')

        maps = MapCollection()

        maps.add_many([
            self.search1,
            self.search2,
        ])

        self.assertEqual(
            maps.ids('search'),
            [
                'search1',
                'search2',
            ],
        )

    def test_map_types(self):
        print('Testing MapCollection: map_types ...')

        maps = MapCollection()

        maps.add_many([
            self.grid,
            self.search1,
        ])

        self.assertEqual(
            maps.map_types(),
            [
                'grid',
                'search',
            ],
        )

    def test_all(self):
        print('Testing MapCollection: all ...')

        maps = MapCollection()

        maps.add_many([
            self.grid,
            self.search1,
            self.search2,
        ])

        result = list(
            maps.all()
        )

        self.assertEqual(
            result,
            [
                self.grid,
                self.search1,
                self.search2,
            ],
        )

    def test_find(self):
        print('Testing MapCollection: find ...')

        maps = MapCollection()

        maps.add_many([
            self.search1,
            self.search2,
        ])

        result = maps.find(
            map_type='search',
            serialem_item='MapFile',
            regex=r'L(\d{2})',
            target_value='02',
        )

        self.assertIs(
            result,
            self.search2,
        )

    def test_find_missing(self):
        print('Testing MapCollection: find missing ...')

        maps = MapCollection()
        maps.add(self.search1)

        result = maps.find(
            map_type='search',
            serialem_item='MapFile',
            regex=r'L(\d{2})',
            target_value='99',
        )

        self.assertIsNone(result)

    def test_find_by_filepath(self):
        print(
            'Testing MapCollection: '
            'find_by_filepath ...'
        )

        maps = MapCollection()

        maps.add_many([
            self.grid,
            self.search1,
        ])

        result = maps.find_by_filepath(
            Path('L01_search.mrc'),
        )

        self.assertIs(
            result,
            self.search1,
        )

    def test_contains(self):
        print('Testing MapCollection: contains ...')

        maps = MapCollection()
        maps.add(self.grid)

        self.assertIn(
            ('grid', 'grid1'),
            maps,
        )

        self.assertNotIn(
            ('grid', 'missing'),
            maps,
        )

    def test_iter(self):
        print('Testing MapCollection: iter ...')

        maps = MapCollection()

        maps.add_many([
            self.grid,
            self.search1,
            self.search2,
        ])

        self.assertEqual(
            list(maps),
            [
                self.grid,
                self.search1,
                self.search2,
            ],
        )

    def test_len(self):
        print('Testing MapCollection: len ...')

        maps = MapCollection()

        self.assertEqual(
            len(maps),
            0,
        )

        maps.add_many([
            self.grid,
            self.search1,
            self.search2,
        ])

        self.assertEqual(
            len(maps),
            3,
        )


class TestMatchRule(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter(
            'ignore',
            category=Warning,
        )

    def test_match_direct(self):
        print('Testing MatchRule: direct match ...')

        rule = MatchRule(
            item_key='MapFile',
            regex=r'L(\d{2})',
        )

        item = {
            'MapFile': 'L03_tgt_012_view.mrc',
        }

        result = rule.match(item)

        self.assertEqual(
            result,
            '03',
        )

    def test_match_multiple_groups(self):
        print('Testing MatchRule: multiple groups ...')

        rule = MatchRule(
            item_key='MapFile',
            regex=r'L(\d{2})_tgt_(\d{3})',
        )

        item = {
            'MapFile': 'L03_tgt_012_view.mrc',
        }

        result = rule.match(item)

        self.assertEqual(
            result,
            '03012',
        )

    def test_match_missing_item_key(self):
        print('Testing MatchRule: missing item key ...')

        rule = MatchRule(
            item_key='MapFile',
            regex=r'L(\d{2})',
        )

        result = rule.match({})

        self.assertIsNone(result)

    def test_match_no_regex_match(self):
        print('Testing MatchRule: no regex match ...')

        rule = MatchRule(
            item_key='MapFile',
            regex=r'L(\d{2})',
        )

        item = {
            'MapFile': 'grid.mrc',
        }

        result = rule.match(item)

        self.assertIsNone(result)

    def test_match_secondary(self):
        print('Testing MatchRule: secondary match ...')

        rule = MatchRule(
            item_key='DrawnID',
            regex=r'(\d+)',
            secondary_item_key='MapFile',
            secondary_regex=r'L(\d{2})',
        )

        item = {
            'DrawnID': '123',
        }

        nav_dict = {
            'items': {
                '123': {
                    'MapFile': 'L07_view.mrc',
                },
            },
        }

        result = rule.match(
            item,
            nav_dict=nav_dict,
        )

        self.assertEqual(
            result,
            '07',
        )

    def test_match_secondary_requires_nav_dict(self):
        print(
            'Testing MatchRule: '
            'secondary match requires nav_dict ...'
        )

        rule = MatchRule(
            item_key='DrawnID',
            regex=r'(\d+)',
            secondary_item_key='MapFile',
            secondary_regex=r'L(\d{2})',
        )

        item = {
            'DrawnID': '123',
        }

        with self.assertRaises(ValueError):
            rule.match(item)

    def test_match_secondary_missing_reference(self):
        print(
            'Testing MatchRule: '
            'secondary missing reference ...'
        )

        rule = MatchRule(
            item_key='DrawnID',
            regex=r'(\d+)',
            secondary_item_key='MapFile',
            secondary_regex=r'L(\d{2})',
        )

        item = {
            'DrawnID': '123',
        }

        nav_dict = {
            'items': {},
        }

        result = rule.match(
            item,
            nav_dict=nav_dict,
        )

        self.assertIsNone(result)

    def test_match_incomplete_secondary_rule(self):
        print(
            'Testing MatchRule: '
            'incomplete secondary rule ...'
        )

        rule = MatchRule(
            item_key='DrawnID',
            regex=r'(\d+)',
            secondary_item_key='MapFile',
        )

        item = {
            'DrawnID': '123',
        }

        nav_dict = {
            'items': {},
        }

        with self.assertRaises(ValueError):
            rule.match(
                item,
                nav_dict=nav_dict,
            )

    def test_from_config_two_entries(self):
        print('Testing MatchRule: from_config two entries ...')

        rule = MatchRule.from_config([
            'MapID',
            r'(.*)',
        ])

        self.assertEqual(
            rule,
            MatchRule(
                item_key='MapID',
                regex=r'(.*)',
            ),
        )

    def test_from_config_four_entries(self):
        print('Testing MatchRule: from_config four entries ...')

        rule = MatchRule.from_config([
            'Note',
            r'^(\d{1,2})',
            'DrawnID',
            r'(.*)',
        ])

        self.assertEqual(
            rule,
            MatchRule(
                item_key='Note',
                regex=r'^(\d{1,2})',
                secondary_item_key='DrawnID',
                secondary_regex=r'(.*)',
            ),
        )

    def test_from_config_match_rule(self):
        print('Testing MatchRule: from_config MatchRule ...')

        original = MatchRule(
            item_key='MapID',
            regex=r'(.*)',
        )

        result = MatchRule.from_config(
            original,
        )

        self.assertIs(
            result,
            original,
        )

    def test_from_config_invalid_length(self):
        print('Testing MatchRule: from_config invalid length ...')

        with self.assertRaises(ValueError):
            MatchRule.from_config([
                'MapID',
                r'(.*)',
                'DrawnID',
            ])

class TestMapHierarchy(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter(
            'ignore',
            category=Warning,
        )

        self.hierarchy = MapHierarchy([
            'grid',
            'search',
            'view',
            'record',
        ])

    def test_init(self):
        print('Testing MapHierarchy: init ...')

        self.assertEqual(
            self.hierarchy.map_type_order,
            [
                'grid',
                'search',
                'view',
                'record',
            ],
        )

    def test_map_type_order_returns_copy(self):
        print(
            'Testing MapHierarchy: '
            'map_type_order returns copy ...'
        )

        result = self.hierarchy.map_type_order
        result.append('tomo')

        self.assertEqual(
            self.hierarchy.map_type_order,
            [
                'grid',
                'search',
                'view',
                'record',
            ],
        )

    def test_add_map_type(self):
        print('Testing MapHierarchy: add_map_type ...')

        hierarchy = MapHierarchy([
            'grid',
            'search',
        ])

        hierarchy.add_map_type('view')

        self.assertEqual(
            hierarchy.map_type_order,
            [
                'grid',
                'search',
                'view',
            ],
        )

    def test_add_duplicate_map_type(self):
        print(
            'Testing MapHierarchy: '
            'add duplicate map type ...'
        )

        with self.assertRaises(ValueError):
            self.hierarchy.add_map_type('view')

    def test_parent_type(self):
        print('Testing MapHierarchy: parent_type ...')

        self.assertIsNone(
            self.hierarchy.parent_type('grid')
        )

        self.assertEqual(
            self.hierarchy.parent_type('search'),
            'grid',
        )

        self.assertEqual(
            self.hierarchy.parent_type('record'),
            'view',
        )

    def test_child_type(self):
        print('Testing MapHierarchy: child_type ...')

        self.assertEqual(
            self.hierarchy.child_type('grid'),
            'search',
        )

        self.assertEqual(
            self.hierarchy.child_type('view'),
            'record',
        )

        self.assertIsNone(
            self.hierarchy.child_type('record')
        )

    def test_unknown_map_type(self):
        print(
            'Testing MapHierarchy: '
            'unknown map type ...'
        )

        with self.assertRaises(ValueError):
            self.hierarchy.parent_type('tomo')

        with self.assertRaises(ValueError):
            self.hierarchy.child_type('tomo')

    def test_add_relation(self):
        print('Testing MapHierarchy: add_relation ...')

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search1',
        )

        self.assertEqual(
            self.hierarchy.parent_id(
                'search',
                'search1',
            ),
            'grid1',
        )

        self.assertEqual(
            self.hierarchy.children_ids(
                'grid',
                'grid1',
            ),
            ['search1'],
        )

    def test_add_multiple_children_preserves_order(self):
        print(
            'Testing MapHierarchy: '
            'child order ...'
        )

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search1',
        )

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search2',
        )

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search3',
        )

        self.assertEqual(
            self.hierarchy.children_ids(
                'grid',
                'grid1',
            ),
            [
                'search1',
                'search2',
                'search3',
            ],
        )

    def test_add_same_relation_twice(self):
        print(
            'Testing MapHierarchy: '
            'add same relation twice ...'
        )

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search1',
        )

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search1',
        )

        self.assertEqual(
            self.hierarchy.children_ids(
                'grid',
                'grid1',
            ),
            ['search1'],
        )

    def test_child_cannot_have_multiple_parents(self):
        print(
            'Testing MapHierarchy: '
            'multiple parents ...'
        )

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search1',
        )

        with self.assertRaises(ValueError):
            self.hierarchy.add_relation(
                'grid',
                'grid2',
                'search',
                'search1',
            )

    def test_invalid_relation_type(self):
        print(
            'Testing MapHierarchy: '
            'invalid relation type ...'
        )

        with self.assertRaises(ValueError):
            self.hierarchy.add_relation(
                'grid',
                'grid1',
                'view',
                'view1',
            )

    def test_remove_relation(self):
        print('Testing MapHierarchy: remove_relation ...')

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search1',
        )

        self.hierarchy.remove_relation(
            'search',
            'search1',
        )

        self.assertIsNone(
            self.hierarchy.parent_id(
                'search',
                'search1',
            )
        )

        self.assertEqual(
            self.hierarchy.children_ids(
                'grid',
                'grid1',
            ),
            [],
        )

    def test_remove_missing_relation(self):
        print(
            'Testing MapHierarchy: '
            'remove missing relation ...'
        )

        with self.assertRaises(KeyError):
            self.hierarchy.remove_relation(
                'search',
                'search1',
            )

    def test_parent_id_root(self):
        print(
            'Testing MapHierarchy: '
            'parent_id root ...'
        )

        self.assertIsNone(
            self.hierarchy.parent_id(
                'grid',
                'grid1',
            )
        )

    def test_children_ids_no_children(self):
        print(
            'Testing MapHierarchy: '
            'children_ids no children ...'
        )

        self.assertEqual(
            self.hierarchy.children_ids(
                'record',
                'record1',
            ),
            [],
        )

    def test_siblings_ids(self):
        print('Testing MapHierarchy: siblings_ids ...')

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search1',
        )

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search2',
        )

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search3',
        )

        self.assertEqual(
            self.hierarchy.siblings_ids(
                'search',
                'search2',
            ),
            [
                'search1',
                'search3',
            ],
        )

    def test_siblings_ids_no_parent(self):
        print(
            'Testing MapHierarchy: '
            'siblings_ids no parent ...'
        )

        self.assertEqual(
            self.hierarchy.siblings_ids(
                'grid',
                'grid1',
            ),
            [],
        )

    def test_ancestors(self):
        print('Testing MapHierarchy: ancestors ...')

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search1',
        )

        self.hierarchy.add_relation(
            'search',
            'search1',
            'view',
            'view1',
        )

        self.hierarchy.add_relation(
            'view',
            'view1',
            'record',
            'record1',
        )

        self.assertEqual(
            self.hierarchy.ancestors(
                'record',
                'record1',
            ),
            [
                ('view', 'view1'),
                ('search', 'search1'),
                ('grid', 'grid1'),
            ],
        )

    def test_ancestors_root(self):
        print('Testing MapHierarchy: ancestors root ...')

        self.assertEqual(
            self.hierarchy.ancestors(
                'grid',
                'grid1',
            ),
            [],
        )

    def test_descendants(self):
        print('Testing MapHierarchy: descendants ...')

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search1',
        )

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search2',
        )

        self.hierarchy.add_relation(
            'search',
            'search1',
            'view',
            'view11',
        )

        self.hierarchy.add_relation(
            'search',
            'search1',
            'view',
            'view12',
        )

        self.hierarchy.add_relation(
            'view',
            'view11',
            'record',
            'record111',
        )

        self.hierarchy.add_relation(
            'view',
            'view12',
            'record',
            'record121',
        )

        self.assertEqual(
            self.hierarchy.descendants(
                'grid',
                'grid1',
            ),
            [
                ('search', 'search1'),
                ('view', 'view11'),
                ('record', 'record111'),
                ('view', 'view12'),
                ('record', 'record121'),
                ('search', 'search2'),
            ],
        )

    def test_descendants_leaf(self):
        print('Testing MapHierarchy: descendants leaf ...')

        self.assertEqual(
            self.hierarchy.descendants(
                'record',
                'record1',
            ),
            [],
        )

    def test_roots(self):
        print('Testing MapHierarchy: roots ...')

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search1',
        )

        self.hierarchy.add_relation(
            'grid',
            'grid2',
            'search',
            'search2',
        )

        self.hierarchy.add_relation(
            'search',
            'search1',
            'view',
            'view1',
        )

        self.assertEqual(
            self.hierarchy.roots(),
            [
                ('grid', 'grid1'),
                ('grid', 'grid2'),
            ],
        )

    def test_validate(self):
        print('Testing MapHierarchy: validate ...')

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search1',
        )

        self.hierarchy.add_relation(
            'search',
            'search1',
            'view',
            'view1',
        )

        self.hierarchy.add_relation(
            'view',
            'view1',
            'record',
            'record1',
        )

        self.hierarchy.validate()

    def test_validate_duplicate_map_types(self):
        print(
            'Testing MapHierarchy: '
            'validate duplicate map types ...'
        )

        hierarchy = MapHierarchy([
            'grid',
            'search',
            'search',
        ])

        with self.assertRaises(ValueError):
            hierarchy.validate()

    def test_validate_inconsistent_parent_lookup(self):
        print(
            'Testing MapHierarchy: '
            'validate inconsistent parent lookup ...'
        )

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search1',
        )

        # Deliberately corrupt the internal state.
        self.hierarchy._parent[
            ('search', 'search1')
        ] = (
            'grid',
            'grid2',
        )

        with self.assertRaises(ValueError):
            self.hierarchy.validate()

    def test_validate_inconsistent_children_lookup(self):
        print(
            'Testing MapHierarchy: '
            'validate inconsistent children lookup ...'
        )

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search1',
        )

        # Deliberately corrupt the internal state.
        self.hierarchy._children[
            ('grid', 'grid1')
        ] = []

        with self.assertRaises(ValueError):
            self.hierarchy.validate()

    def test_iter_depth_first(self):
        print('Testing MapHierarchy: iter_depth_first ...')

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search1',
        )

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search2',
        )

        self.hierarchy.add_relation(
            'search',
            'search1',
            'view',
            'view11',
        )

        self.hierarchy.add_relation(
            'search',
            'search1',
            'view',
            'view12',
        )

        self.hierarchy.add_relation(
            'search',
            'search1',
            'view',
            'view13',
        )

        self.hierarchy.add_relation(
            'view',
            'view11',
            'record',
            'record111',
        )

        self.hierarchy.add_relation(
            'view',
            'view12',
            'record',
            'record121',
        )

        self.hierarchy.add_relation(
            'view',
            'view13',
            'record',
            'record131',
        )

        self.hierarchy.add_relation(
            'search',
            'search2',
            'view',
            'view21',
        )

        self.hierarchy.add_relation(
            'search',
            'search2',
            'view',
            'view22',
        )

        self.hierarchy.add_relation(
            'view',
            'view21',
            'record',
            'record211',
        )

        self.hierarchy.add_relation(
            'view',
            'view22',
            'record',
            'record221',
        )

        result = list(
            self.hierarchy.iter_depth_first()
        )

        expected = [
            ('grid', 'grid1', [0]),

            ('search', 'search1', [0, 0]),

            ('view', 'view11', [0, 0, 0]),
            ('record', 'record111', [0, 0, 0, 0]),

            ('view', 'view12', [0, 0, 1]),
            ('record', 'record121', [0, 0, 1, 0]),

            ('view', 'view13', [0, 0, 2]),
            ('record', 'record131', [0, 0, 2, 0]),

            ('search', 'search2', [0, 1]),

            ('view', 'view21', [0, 1, 0]),
            ('record', 'record211', [0, 1, 0, 0]),

            ('view', 'view22', [0, 1, 1]),
            ('record', 'record221', [0, 1, 1, 0]),
        ]

        self.assertEqual(
            result,
            expected,
        )

    def test_iter_depth_first_multiple_roots(self):
        print(
            'Testing MapHierarchy: '
            'iter_depth_first multiple roots ...'
        )

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search1',
        )

        self.hierarchy.add_relation(
            'grid',
            'grid2',
            'search',
            'search2',
        )

        result = list(
            self.hierarchy.iter_depth_first()
        )

        expected = [
            ('grid', 'grid1', [0]),
            ('search', 'search1', [0, 0]),

            ('grid', 'grid2', [1]),
            ('search', 'search2', [1, 0]),
        ]

        self.assertEqual(
            result,
            expected,
        )

    def test_iter_depth_first_from_root(self):
        print(
            'Testing MapHierarchy: '
            'iter_depth_first from root ...'
        )

        self.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search1',
        )

        self.hierarchy.add_relation(
            'search',
            'search1',
            'view',
            'view11',
        )

        self.hierarchy.add_relation(
            'search',
            'search1',
            'view',
            'view12',
        )

        self.hierarchy.add_relation(
            'view',
            'view11',
            'record',
            'record111',
        )

        result = list(
            self.hierarchy.iter_depth_first(
                root_type='search',
                root_id='search1',
            )
        )

        expected = [
            ('search', 'search1', [0]),
            ('view', 'view11', [0, 0]),
            ('record', 'record111', [0, 0, 0]),
            ('view', 'view12', [0, 1]),
        ]

        self.assertEqual(
            result,
            expected,
        )

    def test_iter_depth_first_incomplete_root(self):
        print(
            'Testing MapHierarchy: '
            'iter_depth_first incomplete root ...'
        )

        with self.assertRaises(ValueError):
            list(
                self.hierarchy.iter_depth_first(
                    root_type='grid',
                )
            )

    def test_iter_depth_first_unknown_root(self):
        print(
            'Testing MapHierarchy: '
            'iter_depth_first unknown root ...'
        )

        with self.assertRaises(KeyError):
            list(
                self.hierarchy.iter_depth_first(
                    root_type='grid',
                    root_id='missing',
                )
            )

    def test_iter_depth_first_isolated_root(self):
        print(
            'Testing MapHierarchy: '
            'iter_depth_first isolated root ...'
        )

        self.hierarchy.add_node(
            'grid',
            'grid1',
        )

        result = list(
            self.hierarchy.iter_depth_first()
        )

        self.assertEqual(
            result,
            [
                ('grid', 'grid1', [0]),
            ],
        )

    def test_roots_includes_isolated_nodes(self):
        print(
            'Testing MapHierarchy: '
            'roots includes isolated nodes ...'
        )

        self.hierarchy.add_node(
            'grid',
            'grid1',
        )

        self.hierarchy.add_node(
            'grid',
            'grid2',
        )

        self.assertEqual(
            self.hierarchy.roots(),
            [
                ('grid', 'grid1'),
                ('grid', 'grid2'),
            ],
        )


class TestNavigatorProfile(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter(
            'ignore',
            category=Warning,
        )

    def test_defaults(self):
        print('Testing NavigatorProfile: defaults ...')

        profile = SingleParticleProfile()

        self.assertEqual(
            profile.map_types,
            [
                'grid',
                'search',
                'view',
                'record',
            ],
        )

        self.assertEqual(
            profile.search_strings,
            {
                'grid': 'gridmap.st',
                'search': '*_search.mrc',
                'view': '*_view.mrc',
                'record': '*_record.mrc',
            },
        )

        self.assertEqual(
            profile.map_binnings,
            {
                'grid': 8,
                'search': 4,
                'view': 1,
                'record': 1,
            },
        )

        self.assertIsNone(
            profile.stitched_dirpath
        )

    def test_subset_map_types(self):
        print(
            'Testing NavigatorProfile: '
            'subset map types ...'
        )

        profile = SingleParticleProfile(
            map_types=[
                'grid',
                'search',
            ]
        )

        self.assertEqual(
            profile.map_types,
            [
                'grid',
                'search',
            ],
        )

        self.assertEqual(
            profile.search_strings,
            {
                'grid': 'gridmap.st',
                'search': '*_search.mrc',
            },
        )

        self.assertEqual(
            profile.map_binnings,
            {
                'grid': 8,
                'search': 4,
            },
        )

    def test_custom_search_strings(self):
        print(
            'Testing NavigatorProfile: '
            'custom search strings ...'
        )

        profile = SingleParticleProfile(
            map_types=[
                'grid',
                'search',
            ],
            search_strings={
                'grid': None,
                'search': '*_custom.mrc',
            },
        )

        self.assertEqual(
            profile.search_strings,
            {
                'grid': None,
                'search': '*_custom.mrc',
            },
        )

    def test_custom_map_binnings(self):
        print(
            'Testing NavigatorProfile: '
            'custom map binnings ...'
        )

        profile = SingleParticleProfile(
            map_types=[
                'grid',
                'search',
            ],
            map_binnings={
                'grid': 4,
                'search': 2,
            },
        )

        self.assertEqual(
            profile.map_binnings,
            {
                'grid': 4,
                'search': 2,
            },
        )

    def test_stitched_dirpath(self):
        print(
            'Testing NavigatorProfile: '
            'stitched_dirpath ...'
        )

        profile = SingleParticleProfile(
            stitched_dirpath='stitched',
        )

        self.assertEqual(
            profile.stitched_dirpath,
            Path('stitched'),
        )

    def test_properties_return_copies(self):
        print(
            'Testing NavigatorProfile: '
            'properties return copies ...'
        )

        profile = SingleParticleProfile()

        map_types = profile.map_types
        map_types.append('other')

        search_strings = profile.search_strings
        search_strings['grid'] = 'changed'

        map_binnings = profile.map_binnings
        map_binnings['grid'] = 100

        self.assertNotIn(
            'other',
            profile.map_types,
        )

        self.assertEqual(
            profile.search_strings['grid'],
            'gridmap.st',
        )

        self.assertEqual(
            profile.map_binnings['grid'],
            8,
        )

    def test_invalid_map_type(self):
        print(
            'Testing NavigatorProfile: '
            'invalid map type ...'
        )

        with self.assertRaises(ValueError):
            SingleParticleProfile(
                map_types=[
                    'grid',
                    'invalid',
                ]
            )

    def test_duplicate_map_types(self):
        print(
            'Testing NavigatorProfile: '
            'duplicate map types ...'
        )

        with self.assertRaises(ValueError):
            SingleParticleProfile(
                map_types=[
                    'grid',
                    'grid',
                ]
            )

    def test_missing_search_string(self):
        print(
            'Testing NavigatorProfile: '
            'missing search string ...'
        )

        with self.assertRaises(ValueError):
            SingleParticleProfile(
                map_types=[
                    'grid',
                    'search',
                ],
                search_strings={
                    'grid': 'gridmap.st',
                },
            )

    def test_extra_search_string(self):
        print(
            'Testing NavigatorProfile: '
            'extra search string ...'
        )

        with self.assertRaises(ValueError):
            SingleParticleProfile(
                map_types=[
                    'grid',
                ],
                search_strings={
                    'grid': 'gridmap.st',
                    'search': '*_search.mrc',
                },
            )

    def test_missing_map_binning(self):
        print(
            'Testing NavigatorProfile: '
            'missing map binning ...'
        )

        with self.assertRaises(ValueError):
            SingleParticleProfile(
                map_types=[
                    'grid',
                    'search',
                ],
                map_binnings={
                    'grid': 8,
                },
            )

    def test_invalid_map_binning(self):
        print(
            'Testing NavigatorProfile: '
            'invalid map binning ...'
        )

        with self.assertRaises(ValueError):
            SingleParticleProfile(
                map_types=[
                    'grid',
                ],
                map_binnings={
                    'grid': 0,
                },
            )

    def test_non_integer_map_binning(self):
        print(
            'Testing NavigatorProfile: '
            'non-integer map binning ...'
        )

        with self.assertRaises(TypeError):
            SingleParticleProfile(
                map_types=[
                    'grid',
                ],
                map_binnings={
                    'grid': 2.5,
                },
            )

    def test_match_rules_from_config(self):
        print(
            'Testing NavigatorProfile: '
            'match rules from config ...'
        )

        profile = SingleParticleProfile(
            match_rules={
                'view': {
                    'search': [
                        'MapID',
                        r'(.*)',
                    ],
                    'view': [
                        'Note',
                        r'^(\d{1,2})',
                        'DrawnID',
                        r'(.*)',
                    ],
                },
            },
        )

        rules = profile.match_rules

        self.assertEqual(
            rules['view']['search'],
            MatchRule(
                item_key='MapID',
                regex=r'(.*)',
            ),
        )

        self.assertEqual(
            rules['view']['view'],
            MatchRule(
                item_key='Note',
                regex=r'^(\d{1,2})',
                secondary_item_key='DrawnID',
                secondary_regex=r'(.*)',
            ),
        )

        
class DummyNavigator:

    def __init__(
        self,
        filepath,
        nav_dict,
    ):
        self.filepath = Path(filepath)
        self.nav_dict = nav_dict


class TestNavigatorProfileDiscovery(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter(
            'ignore',
            category=Warning,
        )

    def test_discover_items(self):
        print(
            'Testing NavigatorProfile: '
            'discover_items ...'
        )

        nav_dict = {
            'items': {
                '1': {
                    'MapFile': 'gridmap.st',
                },
                '2': {
                    'MapFile': 'L01_search.mrc',
                },
                '3': {
                    'MapFile': 'L02_search.mrc',
                },
                '4': {
                    'MapFile': 'L01_view.mrc',
                },
            },
        }

        navigator = DummyNavigator(
            filepath='/data/test.nav',
            nav_dict=nav_dict,
        )

        profile = SingleParticleProfile()

        result = profile.discover_items(
            navigator,
            'search',
        )

        self.assertEqual(
            list(result.keys()),
            [
                '2',
                '3',
            ],
        )


    def test_discover_items_inactive_map_type(self):
        print(
            'Testing NavigatorProfile: '
            'discover_items inactive map type ...'
        )

        navigator = DummyNavigator(
            filepath='/data/test.nav',
            nav_dict={'items': {}},
        )

        profile = SingleParticleProfile(
            map_types=[
                'grid',
                'search',
            ],
        )

        with self.assertRaises(ValueError):
            profile.discover_items(
                navigator,
                'view',
            )

    def test_discover_items_none_search_string(self):
        print(
            'Testing NavigatorProfile: '
            'discover_items None search string ...'
        )

        class TestProfile(NavigatorProfile):
            MAP_TYPES = ('grid',)
            SEARCH_STRINGS = {
                'grid': None,
            }
            MAP_BINNINGS = {
                'grid': 1,
            }

        navigator = DummyNavigator(
            filepath='/data/test.nav',
            nav_dict={'items': {}},
        )

        profile = TestProfile()

        with self.assertRaises(NotImplementedError):
            profile.discover_items(
                navigator,
                'grid',
            )

    def test_source_filepath(self):
        print(
            'Testing NavigatorProfile: '
            'source_filepath ...'
        )

        navigator = DummyNavigator(
            filepath='/data/test.nav',
            nav_dict={},
        )

        profile = SingleParticleProfile()

        item = {
            'MapFile': r'Z:\original\data\L01_view.mrc',
        }

        result = profile.source_filepath(
            navigator,
            'view',
            '1',
            item,
        )

        self.assertEqual(
            result,
            Path('/data/L01_view.mrc'),
        )


    def test_resolve_filepath(self):
        print(
            'Testing NavigatorProfile: '
            'resolve_filepath ...'
        )

        navigator = DummyNavigator(
            filepath='/data/test.nav',
            nav_dict={},
        )

        profile = SingleParticleProfile()

        source_filepath = Path(
            '/data/L01_view.mrc'
        )

        result = NavigatorProfile.resolve_filepath(
            profile,
            navigator,
            'view',
            '1',
            {},
            source_filepath,
        )

        self.assertEqual(
            result,
            source_filepath,
        )


    def test_get_section_id(self):
        print(
            'Testing NavigatorProfile: '
            'get_section_id ...'
        )

        navigator = DummyNavigator(
            filepath='/data/test.nav',
            nav_dict={},
        )

        profile = SingleParticleProfile()

        map_ = NavigatorMap(
            id='1',
            map_type='search',
            serialem_item={
                'Note': 'Sec 42 some description',
            },
            source_filepath='source.mrc',
            filepath='map.mrc',
        )

        result = profile.get_section_id(
            navigator,
            map_,
        )

        self.assertEqual(
            result,
            42,
        )


    def test_get_section_id_missing(self):
        print(
            'Testing NavigatorProfile: '
            'get_section_id missing ...'
        )

        navigator = DummyNavigator(
            filepath='/data/test.nav',
            nav_dict={},
        )

        profile = SingleParticleProfile()

        map_ = NavigatorMap(
            id='1',
            map_type='search',
            serialem_item={
                'Note': 'No section here',
            },
            source_filepath='source.mrc',
            filepath='map.mrc',
        )

        self.assertIsNone(
            profile.get_section_id(
                navigator,
                map_,
            )
        )


    def test_match_id(self):
        print(
            'Testing NavigatorProfile: '
            'match_id ...'
        )

        navigator = DummyNavigator(
            filepath='/data/test.nav',
            nav_dict={'items': {}},
        )

        profile = SingleParticleProfile()

        map_ = NavigatorMap(
            id='1',
            map_type='view',
            serialem_item={
                'MapFile': 'L03_tgt_012_view.mrc',
            },
            source_filepath='source.mrc',
            filepath='map.mrc',
        )

        rule = MatchRule(
            item_key='MapFile',
            regex=r'L(\d{2})_tgt_(\d{3})',
        )

        result = profile.match_id(
            navigator,
            map_,
            rule,
        )

        self.assertEqual(
            result,
            '03012',
        )


class TestProfile(NavigatorProfile):

    MAP_TYPES = (
        'grid',
        'view',
    )

    SEARCH_STRINGS = {
        'grid': '*_grid.mrc',
        'view': '*_view.mrc',
    }

    MAP_BINNINGS = {
        'grid': 2,
        'view': 1,
    }


class TestNavigatorConstruction(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter(
            'ignore',
            category=Warning,
        )

    def _create_mrc(
        self,
        filepath,
        shape=(1, 10, 20),
        voxel_size=(10.0, 10.0, 10.0),
    ):
        import mrcfile

        data = np.zeros(
            shape,
            dtype=np.float32,
        )

        with mrcfile.new(
            filepath,
            overwrite=True,
        ) as mrc:
            mrc.set_data(data)
            mrc.voxel_size = voxel_size

    def test_load_single_navigator_file(self):
        print(
            'Testing Navigator: '
            'load single navigator file ...'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            nav_filepath = tmpdir / 'test.nav'

            nav_filepath.write_text(
                """
AdocVersion = 2.00

[Item = grid]
MapID = 1
MapFile = test_grid.mrc
"""
            )

            self._create_mrc(
                tmpdir / 'test_grid.mrc'
            )

            profile = TestProfile(
                map_types=[
                    'grid',
                ]
            )

            nav = Navigator(
                nav_filepath,
                profile,
            )

        self.assertEqual(
            nav.filepath,
            nav_filepath,
        )

        self.assertEqual(
            nav.all_filepaths,
            [
                nav_filepath,
            ],
        )

        self.assertIn(
            'grid',
            nav.nav_dict['items'],
        )

    def test_load_maps(self):
        print(
            'Testing Navigator: '
            'load maps ...'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            nav_filepath = tmpdir / 'test.nav'

            nav_filepath.write_text(
                """
    AdocVersion = 2.00

    [Item = grid]
    MapID = 1
    MapFile = test_grid.mrc

    [Item = view1]
    MapID = 2
    MapFile = test_view.mrc
    Note = Sec 7 test
    """
            )

            self._create_mrc(
                tmpdir / 'test_grid.mrc',
                shape=(1, 20, 30),
                voxel_size=(10, 10, 10),
            )

            self._create_mrc(
                tmpdir / 'test_view.mrc',
                shape=(1, 40, 50),
                voxel_size=(20, 20, 20),
            )

            profile = TestProfile()

            nav = Navigator(
                nav_filepath,
                profile,
            )

            grid = nav.maps.get(
                'grid',
                'grid',
            )

            view = nav.maps.get(
                'view',
                'view1',
            )

        self.assertEqual(
            len(nav.maps),
            2,
        )

        self.assertEqual(
            grid.binning,
            2,
        )

        self.assertEqual(
            view.binning,
            1,
        )

        self.assertEqual(
            view.section_id,
            7,
        )

        self.assertEqual(
            grid.shape,
            (
                1,
                20,
                30,
            ),
        )

        self.assertTrue(
            np.allclose(
                view.resolution,
                [
                    0.002,
                    0.002,
                    0.002,
                ],
            )
        )

    def test_load_multiple_navigator_files(self):
        print(
            'Testing Navigator: '
            'load multiple navigator files ...'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            nav_filepath_1 = tmpdir / 'test1.nav'
            nav_filepath_2 = tmpdir / 'test2.nav'

            nav_filepath_1.write_text(
                """
    AdocVersion = 2.00

    [Item = grid]
    MapID = 1
    MapFile = test_grid.mrc
    """
            )

            nav_filepath_2.write_text(
                """
    AdocVersion = 2.00

    [Item = view1]
    MapID = 2
    MapFile = test_view.mrc
    """
            )

            self._create_mrc(
                tmpdir / 'test_grid.mrc'
            )

            self._create_mrc(
                tmpdir / 'test_view.mrc'
            )

            profile = TestProfile()

            nav = Navigator(
                [
                    nav_filepath_1,
                    nav_filepath_2,
                ],
                profile,
            )

        self.assertEqual(
            nav.all_filepaths,
            [
                nav_filepath_1,
                nav_filepath_2,
            ],
        )

        self.assertEqual(
            len(nav.maps),
            2,
        )

    def test_load_no_navigator_files(self):
        print(
            'Testing Navigator: '
            'load no navigator files ...'
        )

        profile = TestProfile()

        with self.assertRaises(ValueError):
            Navigator(
                [],
                profile,
            )


class TestNavigatorMapAPI(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter(
            'ignore',
            category=Warning,
        )

        self.nav = Navigator.__new__(Navigator)

        self.nav.maps = MapCollection()

        self.grid = NavigatorMap(
            id='grid1',
            map_type='grid',
            serialem_item={
                'MapFile': 'grid.mrc',
            },
            source_filepath='grid.mrc',
            filepath='grid.mrc',
        )

        self.search1 = NavigatorMap(
            id='search1',
            map_type='search',
            serialem_item={
                'MapFile': 'L01_search.mrc',
            },
            source_filepath='L01_search.mrc',
            filepath='L01_search.mrc',
        )

        self.search2 = NavigatorMap(
            id='search2',
            map_type='search',
            serialem_item={
                'MapFile': 'L02_search.mrc',
            },
            source_filepath='L02_search.mrc',
            filepath='L02_search.mrc',
        )

        self.nav.maps.add_many([
            self.grid,
            self.search1,
            self.search2,
        ])

    def test_get_map(self):
        print('Testing Navigator: get_map ...')

        result = self.nav.get_map(
            'search',
            'search1',
        )

        self.assertIs(
            result,
            self.search1,
        )

    def test_get_maps(self):
        print('Testing Navigator: get_maps ...')

        result = self.nav.get_maps(
            'search',
        )

        self.assertEqual(
            list(result.keys()),
            [
                'search1',
                'search2',
            ],
        )

        self.assertIs(
            result['search1'],
            self.search1,
        )

    def test_get_map_ids(self):
        print('Testing Navigator: get_map_ids ...')

        self.assertEqual(
            self.nav.get_map_ids(
                'search',
            ),
            [
                'search1',
                'search2',
            ],
        )

    def test_get_grid_map(self):
        print('Testing Navigator: get_grid_map ...')

        self.assertIs(
            self.nav.get_grid_map(),
            self.grid,
        )

    def test_get_grid_id(self):
        print('Testing Navigator: get_grid_id ...')

        self.assertEqual(
            self.nav.get_grid_id(),
            'grid1',
        )

    def test_get_grid_map_missing(self):
        print(
            'Testing Navigator: '
            'get_grid_map missing ...'
        )

        nav = Navigator.__new__(
            Navigator
        )
        nav.maps = MapCollection()

        with self.assertRaises(ValueError):
            nav.get_grid_map()

    def test_get_grid_map_multiple(self):
        print(
            'Testing Navigator: '
            'get_grid_map multiple ...'
        )

        nav = Navigator.__new__(
            Navigator
        )
        nav.maps = MapCollection()

        nav.maps.add(
            NavigatorMap(
                id='grid1',
                map_type='grid',
                serialem_item={},
                source_filepath='grid1.mrc',
                filepath='grid1.mrc',
            )
        )

        nav.maps.add(
            NavigatorMap(
                id='grid2',
                map_type='grid',
                serialem_item={},
                source_filepath='grid2.mrc',
                filepath='grid2.mrc',
            )
        )

        with self.assertRaises(ValueError):
            nav.get_grid_map()

    def test_find_item(self):
        print('Testing Navigator: find_item ...')

        result = self.nav.find_item(
            map_type='search',
            serialem_item='MapFile',
            regex=r'L(\d{2})',
            target_value='02',
        )

        self.assertIs(
            result,
            self.search2,
        )

    def test_find_item_missing(self):
        print(
            'Testing Navigator: '
            'find_item missing ...'
        )

        result = self.nav.find_item(
            map_type='search',
            serialem_item='MapFile',
            regex=r'L(\d{2})',
            target_value='99',
        )

        self.assertIsNone(
            result,
        )

    def test_iter_maps(self):
        print('Testing Navigator: iter_maps ...')

        result = list(
            self.nav.iter_maps()
        )

        self.assertEqual(
            result,
            [
                self.grid,
                self.search1,
                self.search2,
            ],
        )

    def test_iter_maps_by_type(self):
        print(
            'Testing Navigator: '
            'iter_maps by type ...'
        )

        result = list(
            self.nav.iter_maps(
                'search',
            )
        )

        self.assertEqual(
            result,
            [
                self.search1,
                self.search2,
            ],
        )

    def test_add_maps_new_map_type(self):
        print(
            'Testing Navigator: '
            'add_maps new map type ...'
        )

        nav = Navigator.__new__(Navigator)

        nav.profile = TomoCLEMProfile()
        nav.maps = MapCollection()
        nav.hierarchy = MapHierarchy(
            nav.profile.map_types
        )

        target = NavigatorMap(
            id='tgt1',
            map_type='tgt',
            serialem_item={},
            source_filepath='tgt1.mrc',
            filepath='tgt1.mrc',
        )

        nav.maps.add(target)
        nav.hierarchy.add_node(
            'tgt',
            'tgt1',
        )

        tomo = NavigatorMap(
            id='tomo1',
            map_type='tomo',
            serialem_item={},
            source_filepath='tomo1.mrc',
            filepath='tomo1.mrc',
        )

        nav.add_maps(
            'tomo',
            [tomo],
            parent_map_type='tgt',
        )

        self.assertIn(
            'tomo',
            nav.map_types,
        )

        self.assertIs(
            nav.get_map(
                'tomo',
                'tomo1',
            ),
            tomo,
        )

        self.assertIn(
            ('tomo', 'tomo1'),
            nav.hierarchy._nodes,
        )

    def test_remove_map_type(self):
        print(
            'Testing Navigator: '
            'remove_map_type ...'
        )

        nav = Navigator.__new__(Navigator)

        nav.profile = TomoCLEMProfile()
        nav.maps = MapCollection()
        nav.hierarchy = MapHierarchy(
            nav.profile.map_types
        )

        target = NavigatorMap(
            id='tgt1',
            map_type='tgt',
            serialem_item={},
            source_filepath='tgt1.mrc',
            filepath='tgt1.mrc',
        )

        nav.maps.add(target)
        nav.hierarchy.add_node(
            'tgt',
            'tgt1',
        )

        tomo = NavigatorMap(
            id='tomo1',
            map_type='tomo',
            serialem_item={},
            source_filepath='tomo1.mrc',
            filepath='tomo1.mrc',
        )

        nav.add_maps(
            'tomo',
            [tomo],
            parent_map_type='tgt',
        )

        nav.hierarchy.add_relation(
            'tgt',
            'tgt1',
            'tomo',
            'tomo1',
        )

        nav.remove_map_type(
            'tomo'
        )

        self.assertNotIn(
            'tomo',
            nav.map_types,
        )

        self.assertEqual(
            nav.get_map_ids('tomo'),
            [],
        )

        self.assertEqual(
            nav.hierarchy.map_type_order,
            [
                'grid',
                'lamella',
                'view',
                'tgt',
            ],
        )
        
class TestSingleParticleProfile(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter(
            'ignore',
            category=Warning,
        )

    def test_discover_search_items(self):
        print(
            'Testing SingleParticleProfile: '
            'discover search items ...'
        )

        nav = DummyNavigator(
            filepath='/data/test.nav',
            nav_dict={
                'items': {
                    '1': {
                        'MapFile': 'gridmap.st',
                    },
                    '2': {
                        'MapFile': 'L01_search.mrc',
                    },
                    '3': {
                        'MapFile': 'L02_search.mrc',
                    },
                    '4': {
                        'MapFile': 'L01_view.mrc',
                    },
                },
            },
        )

        profile = SingleParticleProfile()

        result = profile.discover_items(
            nav,
            'search',
        )

        self.assertEqual(
            list(result),
            [
                '2',
                '3',
            ],
        )

    def test_discover_grid_none_search_string(self):
        print(
            'Testing SingleParticleProfile: '
            'discover grid without search string ...'
        )

        nav = DummyNavigator(
            filepath='/data/test.nav',
            nav_dict={
                'items': {
                    'grid': {
                        'MapFile': 'grid.mrc',
                    },
                },
            },
        )

        profile = SingleParticleProfile(
            search_strings={
                'grid': None,
                'search': '*_search.mrc',
                'view': '*_view.mrc',
                'record': '*_record.mrc',
            },
        )

        result = profile.discover_items(
            nav,
            'grid',
        )

        self.assertEqual(
            list(result),
            ['grid'],
        )

    def test_resolve_grid_filepath(self):
        print(
            'Testing SingleParticleProfile: '
            'resolve grid filepath ...'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_filepath = (
                tmpdir / 'gridmap.st'
            )

            expected = (
                tmpdir
                / 'gridmap_test_bin8.st'
            )

            expected.touch()

            nav = DummyNavigator(
                filepath=tmpdir / 'test.nav',
                nav_dict={},
            )

            profile = SingleParticleProfile()

            result = profile.resolve_grid_filepath(
                nav,
                source_filepath,
                {
                    'Note': 'Grid 01 test',
                },
            )

            self.assertEqual(
                result,
                expected,
            )

    def test_resolve_grid_filepath_mrc_fallback(self):
        print(
            'Testing SingleParticleProfile: '
            'resolve grid filepath MRC fallback ...'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_filepath = (
                tmpdir / 'gridmap.st'
            )

            expected = (
                tmpdir
                / 'gridmap_test_bin8.mrc'
            )

            expected.touch()

            nav = DummyNavigator(
                filepath=tmpdir / 'test.nav',
                nav_dict={},
            )

            profile = SingleParticleProfile()

            result = profile.resolve_grid_filepath(
                nav,
                source_filepath,
                {
                    'Note': 'Grid 01 test',
                },
            )

            self.assertEqual(
                result,
                expected,
            )

    def test_resolve_search_filepath(self):
        print(
            'Testing SingleParticleProfile: '
            'resolve search filepath ...'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_filepath = (
                tmpdir / 'L01_search.mrc'
            )

            expected = (
                tmpdir
                / 'L01_search_0006_bin4.mrc'
            )

            expected.touch()

            nav = DummyNavigator(
                filepath=tmpdir / 'test.nav',
                nav_dict={},
            )

            profile = SingleParticleProfile()

            result = profile.resolve_search_filepath(
                nav,
                source_filepath,
                {
                    'Note': 'Sec 5 something',
                },
            )

            self.assertEqual(
                result,
                expected,
            )

    def test_resolve_view_filepath(self):
        print(
            'Testing SingleParticleProfile: '
            'resolve view filepath ...'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_filepath = (
                tmpdir / 'L01_view.mrc'
            )

            expected = (
                tmpdir
                / 'L01_view_0006.mrc'
            )

            expected.touch()

            nav = DummyNavigator(
                filepath=tmpdir / 'test.nav',
                nav_dict={},
            )

            profile = SingleParticleProfile()

            result = profile.resolve_view_filepath(
                nav,
                source_filepath,
                {
                    'Note': 'Sec 5 something',
                },
            )

            self.assertEqual(
                result,
                expected,
            )

    def test_resolve_record_filepath(self):
        print(
            'Testing SingleParticleProfile: '
            'resolve record filepath ...'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_filepath = (
                tmpdir / 'L01_record.mrc'
            )

            expected = (
                tmpdir
                / 'L01_record_0011.mrc'
            )

            expected.touch()

            nav = DummyNavigator(
                filepath=tmpdir / 'test.nav',
                nav_dict={},
            )

            profile = SingleParticleProfile()

            result = profile.resolve_record_filepath(
                nav,
                source_filepath,
                {
                    'Note': 'Sec 10 something',
                },
            )

            self.assertEqual(
                result,
                expected,
            )


class TestSingleParticleProfileRelationships(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter(
            'ignore',
            category=Warning,
        )

        self.nav = Navigator.__new__(Navigator)
        self.nav.maps = MapCollection()

        self.nav.nav_dict = {
            'items': {
                # Referenced by the two-step view MatchRule.
                '11': {
                    'DrawnID': '1001',
                },
                '12': {
                    'DrawnID': '1002',
                },
            },
        }

        self.grid = NavigatorMap(
            id='grid1',
            map_type='grid',
            serialem_item={
                'MapID': 'grid1',
            },
            source_filepath='grid.mrc',
            filepath='grid.mrc',
        )

        self.search1 = NavigatorMap(
            id='search1',
            map_type='search',
            serialem_item={
                'MapID': '1001',
            },
            source_filepath='search1.mrc',
            filepath='search1.mrc',
        )

        self.search2 = NavigatorMap(
            id='search2',
            map_type='search',
            serialem_item={
                'MapID': '1002',
            },
            source_filepath='search2.mrc',
            filepath='search2.mrc',
        )

        self.view11 = NavigatorMap(
            id='view11',
            map_type='view',
            serialem_item={
                'Note': '11 view',
            },
            source_filepath='view11.mrc',
            filepath='view11.mrc',
        )

        self.view12 = NavigatorMap(
            id='view12',
            map_type='view',
            serialem_item={
                'Note': '12 view',
            },
            source_filepath='view12.mrc',
            filepath='view12.mrc',
        )

        self.record111 = NavigatorMap(
            id='record111',
            map_type='record',
            serialem_item={
                'Note': '11 record',
            },
            source_filepath='record111.mrc',
            filepath='record111.mrc',
        )

        self.record121 = NavigatorMap(
            id='record121',
            map_type='record',
            serialem_item={
                'Note': '12 record',
            },
            source_filepath='record121.mrc',
            filepath='record121.mrc',
        )

        self.nav.maps.add_many([
            self.grid,
            self.search1,
            self.search2,
            self.view11,
            self.view12,
            self.record111,
            self.record121,
        ])

        self.profile = SingleParticleProfile()

    def test_build_relationships(self):
        print(
            'Testing SingleParticleProfile: '
            'build_relationships ...'
        )

        hierarchy = MapHierarchy(
            self.profile.map_types,
        )

        self.profile.build_relationships(
            self.nav,
            hierarchy,
        )

        self.assertEqual(
            hierarchy.parent_id(
                'search',
                'search1',
            ),
            'grid1',
        )

        self.assertEqual(
            hierarchy.parent_id(
                'search',
                'search2',
            ),
            'grid1',
        )

        self.assertEqual(
            hierarchy.parent_id(
                'view',
                'view11',
            ),
            'search1',
        )

        self.assertEqual(
            hierarchy.parent_id(
                'view',
                'view12',
            ),
            'search2',
        )

        self.assertEqual(
            hierarchy.parent_id(
                'record',
                'record111',
            ),
            'view11',
        )

        self.assertEqual(
            hierarchy.parent_id(
                'record',
                'record121',
            ),
            'view12',
        )

    def test_build_relationships_depth_first_order(self):
        print(
            'Testing SingleParticleProfile: '
            'build_relationships depth-first order ...'
        )

        hierarchy = MapHierarchy(
            self.profile.map_types,
        )

        self.profile.build_relationships(
            self.nav,
            hierarchy,
        )

        result = list(
            hierarchy.iter_depth_first()
        )

        expected = [
            ('grid', 'grid1', [0]),

            ('search', 'search1', [0, 0]),
            ('view', 'view11', [0, 0, 0]),
            ('record', 'record111', [0, 0, 0, 0]),

            ('search', 'search2', [0, 1]),
            ('view', 'view12', [0, 1, 0]),
            ('record', 'record121', [0, 1, 0, 0]),
        ]

        self.assertEqual(
            result,
            expected,
        )

    def test_build_relationships_registers_all_nodes(self):
        print(
            'Testing SingleParticleProfile: '
            'build_relationships registers all nodes ...'
        )

        orphan_view = NavigatorMap(
            id='orphan_view',
            map_type='view',
            serialem_item={
                'Note': '99 view',
            },
            source_filepath='orphan_view.mrc',
            filepath='orphan_view.mrc',
        )

        self.nav.maps.add(
            orphan_view,
        )

        self.nav.profile = self.profile

        self.nav._build_hierarchy()

        self.assertIn(
            ('view', 'orphan_view'),
            self.nav.hierarchy._nodes,
        )

        self.assertIsNone(
            self.nav.hierarchy.parent_id(
                'view',
                'orphan_view',
            )
        )

    def test_build_relationships_ambiguous_parent(self):
        print(
            'Testing SingleParticleProfile: '
            'build_relationships ambiguous parent ...'
        )

        duplicate_search = NavigatorMap(
            id='search_duplicate',
            map_type='search',
            serialem_item={
                'MapID': '1001',
            },
            source_filepath='search_duplicate.mrc',
            filepath='search_duplicate.mrc',
        )

        self.nav.maps.add(
            duplicate_search,
        )

        hierarchy = MapHierarchy(
            self.profile.map_types,
        )

        with self.assertRaises(ValueError):
            self.profile.build_relationships(
                self.nav,
                hierarchy,
            )

    def test_build_relationships_requires_single_grid(self):
        print(
            'Testing SingleParticleProfile: '
            'build_relationships requires single grid ...'
        )

        second_grid = NavigatorMap(
            id='grid2',
            map_type='grid',
            serialem_item={},
            source_filepath='grid2.mrc',
            filepath='grid2.mrc',
        )

        self.nav.maps.add(
            second_grid,
        )

        hierarchy = MapHierarchy(
            self.profile.map_types,
        )

        with self.assertRaises(ValueError):
            self.profile.build_relationships(
                self.nav,
                hierarchy,
            )


class TestNavigatorHierarchy(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter(
            'ignore',
            category=Warning,
        )

        self.nav = Navigator.__new__(
            Navigator
        )

        self.nav.maps = MapCollection()

        self.grid = NavigatorMap(
            id='grid1',
            map_type='grid',
            serialem_item={},
            source_filepath='grid.mrc',
            filepath='grid.mrc',
        )

        self.search1 = NavigatorMap(
            id='search1',
            map_type='search',
            serialem_item={},
            source_filepath='search1.mrc',
            filepath='search1.mrc',
        )

        self.search2 = NavigatorMap(
            id='search2',
            map_type='search',
            serialem_item={},
            source_filepath='search2.mrc',
            filepath='search2.mrc',
        )

        self.view11 = NavigatorMap(
            id='view11',
            map_type='view',
            serialem_item={},
            source_filepath='view11.mrc',
            filepath='view11.mrc',
        )

        self.view12 = NavigatorMap(
            id='view12',
            map_type='view',
            serialem_item={},
            source_filepath='view12.mrc',
            filepath='view12.mrc',
        )

        self.record111 = NavigatorMap(
            id='record111',
            map_type='record',
            serialem_item={},
            source_filepath='record111.mrc',
            filepath='record111.mrc',
        )

        self.record121 = NavigatorMap(
            id='record121',
            map_type='record',
            serialem_item={},
            source_filepath='record121.mrc',
            filepath='record121.mrc',
        )

        self.nav.maps.add_many([
            self.grid,
            self.search1,
            self.search2,
            self.view11,
            self.view12,
            self.record111,
            self.record121,
        ])

        self.nav.hierarchy = MapHierarchy([
            'grid',
            'search',
            'view',
            'record',
        ])

        self.nav.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search1',
        )

        self.nav.hierarchy.add_relation(
            'grid',
            'grid1',
            'search',
            'search2',
        )

        self.nav.hierarchy.add_relation(
            'search',
            'search1',
            'view',
            'view11',
        )

        self.nav.hierarchy.add_relation(
            'search',
            'search1',
            'view',
            'view12',
        )

        self.nav.hierarchy.add_relation(
            'view',
            'view11',
            'record',
            'record111',
        )

        self.nav.hierarchy.add_relation(
            'view',
            'view12',
            'record',
            'record121',
        )

    def test_get_parent(self):
        print('Testing Navigator: get_parent ...')

        result = self.nav.get_parent(
            'view',
            'view11',
        )

        self.assertIs(
            result,
            self.search1,
        )

    def test_get_parent_root(self):
        print(
            'Testing Navigator: '
            'get_parent root ...'
        )

        self.assertIsNone(
            self.nav.get_parent(
                'grid',
                'grid1',
            )
        )

    def test_get_children(self):
        print('Testing Navigator: get_children ...')

        result = self.nav.get_children(
            'search',
            'search1',
        )

        self.assertEqual(
            result,
            [
                self.view11,
                self.view12,
            ],
        )

    def test_get_siblings(self):
        print('Testing Navigator: get_siblings ...')

        result = self.nav.get_siblings(
            'view',
            'view11',
        )

        self.assertEqual(
            result,
            [
                self.view12,
            ],
        )

    def test_get_ancestors(self):
        print('Testing Navigator: get_ancestors ...')

        result = self.nav.get_ancestors(
            'record',
            'record111',
        )

        self.assertEqual(
            result,
            [
                self.view11,
                self.search1,
                self.grid,
            ],
        )

    def test_get_ancestor(self):
        print('Testing Navigator: get_ancestor ...')

        result = self.nav.get_ancestor(
            'record',
            'record111',
            ancestor_type='search',
        )

        self.assertIs(
            result,
            self.search1,
        )

    def test_get_ancestor_missing(self):
        print(
            'Testing Navigator: '
            'get_ancestor missing ...'
        )

        result = self.nav.get_ancestor(
            'view',
            'view11',
            ancestor_type='record',
        )

        self.assertIsNone(
            result,
        )

    def test_get_descendants(self):
        print('Testing Navigator: get_descendants ...')

        result = self.nav.get_descendants(
            'grid',
            'grid1',
        )

        self.assertEqual(
            result,
            [
                self.search1,
                self.view11,
                self.record111,
                self.view12,
                self.record121,
                self.search2,
            ],
        )

    def test_iter_hierarchy(self):
        print('Testing Navigator: iter_hierarchy ...')

        result = [
            (
                map_.id,
                idx_path,
            )
            for map_, idx_path
            in self.nav.iter_hierarchy()
        ]

        expected = [
            ('grid1', [0]),
            ('search1', [0, 0]),
            ('view11', [0, 0, 0]),
            ('record111', [0, 0, 0, 0]),
            ('view12', [0, 0, 1]),
            ('record121', [0, 0, 1, 0]),
            ('search2', [0, 1]),
        ]

        self.assertEqual(
            result,
            expected,
        )

    def test_iter(self):
        print('Testing Navigator: __iter__ ...')

        result = [
            (
                map_.id,
                idx_path,
            )
            for map_, idx_path
            in self.nav
        ]

        expected = [
            ('grid1', [0]),
            ('search1', [0, 0]),
            ('view11', [0, 0, 0]),
            ('record111', [0, 0, 0, 0]),
            ('view12', [0, 0, 1]),
            ('record121', [0, 0, 1, 0]),
            ('search2', [0, 1]),
        ]

        self.assertEqual(
            result,
            expected,
        )


class TestNavigatorMetadataAPI(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter(
            'ignore',
            category=Warning,
        )

        self.nav = Navigator.__new__(
            Navigator
        )

        self.nav.maps = MapCollection()

    def test_get_map_resolution(self):
        print(
            'Testing Navigator: '
            'get_map_resolution ...'
        )

        map_ = NavigatorMap(
            id='view1',
            map_type='view',
            serialem_item={},
            source_filepath='view1.mrc',
            filepath='view1.mrc',
            resolution=np.array([
                0.001,
                0.002,
                0.003,
            ]),
        )

        self.nav.maps.add(map_)

        result = self.nav.get_map_resolution(
            'view',
            'view1',
        )

        self.assertTrue(
            np.allclose(
                result,
                [
                    0.001,
                    0.002,
                    0.003,
                ],
            )
        )

    def test_get_map_shape(self):
        print(
            'Testing Navigator: '
            'get_map_shape ...'
        )

        map_ = NavigatorMap(
            id='view1',
            map_type='view',
            serialem_item={},
            source_filepath='view1.mrc',
            filepath='view1.mrc',
            shape=(
                1,
                20,
                30,
            ),
        )

        self.nav.maps.add(map_)

        result = self.nav.get_map_shape(
            'view',
            'view1',
        )

        self.assertEqual(
            result,
            (
                1,
                20,
                30,
            ),
        )

    def test_get_map_section_id(self):
        print(
            'Testing Navigator: '
            'get_map_section_id ...'
        )

        map_ = NavigatorMap(
            id='view1',
            map_type='view',
            serialem_item={},
            source_filepath='view1.mrc',
            filepath='view1.mrc',
            section_id=42,
        )

        self.nav.maps.add(map_)

        result = self.nav.get_map_section_id(
            'view',
            'view1',
        )

        self.assertEqual(
            result,
            42,
        )

    def test_get_map_section_id_missing(self):
        print(
            'Testing Navigator: '
            'get_map_section_id missing ...'
        )

        map_ = NavigatorMap(
            id='view1',
            map_type='view',
            serialem_item={},
            source_filepath='view1.mrc',
            filepath='view1.mrc',
        )

        self.nav.maps.add(map_)

        self.assertIsNone(
            self.nav.get_map_section_id(
                'view',
                'view1',
            )
        )

    def test_get_contrast_limits(self):
        print(
            'Testing Navigator: '
            'get_contrast_limits ...'
        )

        map1 = NavigatorMap(
            id='view1',
            map_type='view',
            serialem_item={},
            source_filepath='view1.mrc',
            filepath='view1.mrc',
            contrast_limits=(
                10.0,
                100.0,
            ),
        )

        map2 = NavigatorMap(
            id='view2',
            map_type='view',
            serialem_item={},
            source_filepath='view2.mrc',
            filepath='view2.mrc',
            contrast_limits=(
                20.0,
                200.0,
            ),
        )

        self.nav.maps.add_many([
            map1,
            map2,
        ])

        result = self.nav.get_contrast_limits(
            'view',
        )

        self.assertEqual(
            result,
            {
                'view1': (
                    10.0,
                    100.0,
                ),
                'view2': (
                    20.0,
                    200.0,
                ),
            },
        )


class TestNavigatorAffineAPI(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter(
            'ignore',
            category=Warning,
        )

        self.nav = Navigator.__new__(
            Navigator
        )

        self.nav.maps = MapCollection()

        self.grid = NavigatorMap(
            id='grid1',
            map_type='grid',
            serialem_item={
                'StageXYZ': '10 20 0',
                'MapScaleMat': '1 0 0 1',
            },
            source_filepath='grid.mrc',
            filepath='grid.mrc',
            shape=(1, 100, 200),
            resolution=np.array([
                1.0,
                1.0,
                1.0,
            ]),
        )

        self.nav.maps.add(
            self.grid
        )

    def test_get_map_affine(self):
        print(
            'Testing Navigator: '
            'get_map_affine ...'
        )

        result = self.nav.get_map_affine(
            'grid',
            'grid1',
            full_square=True,
        )

        expected = np.array([
            [1, 0, 0, 90],
            [0, 1, 0, 30],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)

        self.assertTrue(
            np.allclose(
                result,
                expected,
            )
        )

    def test_get_map_affine_invert(self):
        print(
            'Testing Navigator: '
            'get_map_affine invert ...'
        )

        result = self.nav.get_map_affine(
            'grid',
            'grid1',
            invert=True,
            full_square=True,
        )

        expected = np.array([
            [1, 0, 0, -90],
            [0, 1, 0, -30],
            [0, 0, 1,   0],
            [0, 0, 0,   1],
        ], dtype=float)

        self.assertTrue(
            np.allclose(
                result,
                expected,
            )
        )

    def test_get_map_affine_output_shape(self):
        print(
            'Testing Navigator: '
            'get_map_affine output shape ...'
        )

        affine = self.nav.get_map_affine(
            'grid',
            'grid1',
        )

        self.assertEqual(
            affine.shape,
            (3, 4),
        )

        affine_square = self.nav.get_map_affine(
            'grid',
            'grid1',
            full_square=True,
        )

        self.assertEqual(
            affine_square.shape,
            (4, 4),
        )

        affine_flat = self.nav.get_map_affine(
            'grid',
            'grid1',
            full_square=True,
            flatten=True,
        )

        self.assertEqual(
            affine_flat.shape,
            (16,),
        ) 

    def test_get_map_full_affines_grid_identity(self):
        print(
            'Testing Navigator: '
            'get_map_full_affines grid identity ...'
        )

        result = self.nav.get_map_full_affines(
            'grid',
            stage_coordinate_system=False,
        )

        affine = result[
            'grid1'
        ].reshape(4, 4)

        self.assertTrue(
            np.allclose(
                affine,
                np.eye(4),
                atol=1e-6,
            )
        )

    def test_get_map_full_affines_relative_to_grid(self):
        print(
            'Testing Navigator: '
            'get_map_full_affines relative to grid ...'
        )

        search = NavigatorMap(
            id='search1',
            map_type='search',
            serialem_item={
                'StageXYZ': '20 30 0',
                'MapScaleMat': '1 0 0 1',
            },
            source_filepath='search.mrc',
            filepath='search.mrc',
            shape=(1, 100, 200),
            resolution=np.array([
                1.0,
                1.0,
                1.0,
            ]),
        )

        self.nav.maps.add(
            search
        )

        result = self.nav.get_map_full_affines(
            'search',
            stage_coordinate_system=False,
        )

        affine = result[
            'search1'
        ].reshape(4, 4)

        expected = np.eye(4)

        expected[0, 3] = 10
        expected[1, 3] = 10

        self.assertTrue(
            np.allclose(
                affine,
                expected,
                atol=1e-6,
            )
        )

    def test_get_map_affine_anisotropic_xy(self):
        print(
            'Testing Navigator: '
            'get_map_affine anisotropic XY ...'
        )

        self.grid.resolution = np.array([
            1.0,
            2.0,
            1.0,
        ])

        with self.assertRaises(ValueError):
            self.nav.get_map_affine(
                'grid',
                'grid1',
            )


class TestTomoCLEMProfile(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter(
            'ignore',
            category=Warning,
        )

    def test_discover_grid(self):
        print(
            'Testing TomoCLEMProfile: '
            'discover grid ...'
        )

        navigator = DummyNavigator(
            filepath='/data/nav/test.nav',
            nav_dict={
                'items': {
                    'grid1': {
                        'MapFile': 'gridmap.st',
                    },
                    'lamella1': {
                        'MapFile': 'L_01.map',
                    },
                },
            },
        )

        profile = TomoCLEMProfile()

        result = profile.discover_items(
            navigator,
            'grid',
        )

        self.assertEqual(
            list(result),
            ['grid1'],
        )

    def test_discover_lamella(self):
        print(
            'Testing TomoCLEMProfile: '
            'discover lamella ...'
        )

        navigator = DummyNavigator(
            filepath='/data/nav/test.nav',
            nav_dict={
                'items': {
                    '1': {
                        'MapFile': 'L_01.map',
                    },
                    '2': {
                        'MapFile': 'L_02.map',
                    },
                    '3': {
                        'MapFile': 'L01_tgt_001_view.mrc',
                    },
                },
            },
        )

        profile = TomoCLEMProfile()

        result = profile.discover_items(
            navigator,
            'lamella',
        )

        self.assertEqual(
            list(result),
            [
                '1',
                '2',
            ],
        )

    def test_resolve_grid_filepath(self):
        print(
            'Testing TomoCLEMProfile: '
            'resolve grid filepath ...'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_filepath = (
                tmpdir / 'gridmap.st'
            )

            expected = (
                tmpdir
                / 'gridmap_stitched_grid01_bin8.mrc'
            )

            expected.touch()

            navigator = DummyNavigator(
                filepath=tmpdir / 'test.nav',
                nav_dict={},
            )

            profile = TomoCLEMProfile()

            result = profile.resolve_grid_filepath(
                navigator,
                source_filepath,
            )

            self.assertEqual(
                result,
                expected,
            )

    def test_resolve_lamella_filepath(self):
        print(
            'Testing TomoCLEMProfile: '
            'resolve lamella filepath ...'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_filepath = (
                tmpdir / 'L_01.map'
            )

            expected = (
                tmpdir
                / 'L_01_stitched_grid01_bin8.mrc'
            )

            expected.touch()

            navigator = DummyNavigator(
                filepath=tmpdir / 'test.nav',
                nav_dict={},
            )

            profile = TomoCLEMProfile()

            result = profile.resolve_lamella_filepath(
                navigator,
                source_filepath,
            )

            self.assertEqual(
                result,
                expected,
            )

    def test_resolve_view_filepath_same_directory(self):
        print(
            'Testing TomoCLEMProfile: '
            'resolve view same directory ...'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            nav_dir = tmpdir / 'nav'
            nav_dir.mkdir()

            expected = (
                nav_dir
                / 'L01_tgt_001_view.mrc'
            )
            expected.touch()

            navigator = DummyNavigator(
                filepath=nav_dir / 'test.nav',
                nav_dict={},
            )

            profile = TomoCLEMProfile()

            result = profile.resolve_view_filepath(
                navigator,
                Path(
                    '/some/old/path/'
                    'L01_tgt_001_view.mrc'
                ),
            )

            self.assertEqual(
                result,
                expected,
            )

    def test_resolve_view_filepath_pace(self):
        print(
            'Testing TomoCLEMProfile: '
            'resolve view pace ...'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            nav_dir = tmpdir / 'nav'
            pace_dir = tmpdir / 'pace'

            nav_dir.mkdir()
            pace_dir.mkdir()

            expected = (
                pace_dir
                / 'L01_tgt_001_view.mrc'
            )
            expected.touch()

            navigator = DummyNavigator(
                filepath=nav_dir / 'test.nav',
                nav_dict={},
            )

            profile = TomoCLEMProfile()

            result = profile.resolve_view_filepath(
                navigator,
                Path(
                    '/some/old/path/'
                    'L01_tgt_001_view.mrc'
                ),
            )

            self.assertEqual(
                result,
                expected,
            )

    def test_resolve_target_filepath_missing(self):
        print(
            'Testing TomoCLEMProfile: '
            'resolve missing target filepath ...'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            nav_dir = tmpdir / 'nav'
            nav_dir.mkdir()

            navigator = DummyNavigator(
                filepath=nav_dir / 'test.nav',
                nav_dict={},
            )

            profile = TomoCLEMProfile()

            result = profile.resolve_target_filepath(
                navigator,
                Path(
                    '/some/old/path/'
                    'L01_tgt_001.mrc'
                ),
            )

            self.assertEqual(
                result,
                tmpdir
                / 'pace'
                / 'L01_tgt_001.mrc',
            )

    def test_build_relationships(self):
        print(
            'Testing TomoCLEMProfile: '
            'build_relationships ...'
        )

        nav = Navigator.__new__(
            Navigator
        )

        nav.maps = MapCollection()

        grid = NavigatorMap(
            id='grid1',
            map_type='grid',
            serialem_item={
                'MapFile': 'gridmap.st',
            },
            source_filepath='grid.mrc',
            filepath='grid.mrc',
        )

        lamella = NavigatorMap(
            id='lamella1',
            map_type='lamella',
            serialem_item={
                'MapFile': 'L_01.map',
            },
            source_filepath='lamella.mrc',
            filepath='lamella.mrc',
        )

        view = NavigatorMap(
            id='view1',
            map_type='view',
            serialem_item={
                'MapFile': 'L01_tgt_001_view.mrc',
            },
            source_filepath='view.mrc',
            filepath='view.mrc',
        )

        target = NavigatorMap(
            id='target1',
            map_type='tgt',
            serialem_item={
                'MapFile': 'L01_tgt_001.mrc',
            },
            source_filepath='target.mrc',
            filepath='target.mrc',
        )

        nav.maps.add_many([
            grid,
            lamella,
            view,
            target,
        ])

        nav.nav_dict = {
            'items': {},
        }

        profile = TomoCLEMProfile()
        nav.profile = profile

        nav._build_hierarchy()

        self.assertIs(
            nav.get_parent(
                'lamella',
                'lamella1',
            ),
            grid,
        )

        self.assertIs(
            nav.get_parent(
                'view',
                'view1',
            ),
            lamella,
        )

        self.assertIs(
            nav.get_parent(
                'tgt',
                'target1',
            ),
            view,
        )

        result = [
            map_.id
            for map_, idx_path in nav
        ]

        self.assertEqual(
            result,
            [
                'grid1',
                'lamella1',
                'view1',
                'target1',
            ],
        )
        

class TestNavigatorTomograms(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter(
            'ignore',
            category=Warning,
        )

    def _create_mrc(
        self,
        filepath,
        shape=(5, 20, 30),
    ):
        import mrcfile

        data = np.zeros(
            shape,
            dtype=np.float32,
        )

        with mrcfile.new(
            filepath,
            overwrite=True,
        ) as mrc:
            mrc.set_data(data)

    def _create_nav(self):

        nav = Navigator.__new__(
            Navigator
        )

        nav.profile = TomoCLEMProfile()
        nav.maps = MapCollection()

        nav.hierarchy = MapHierarchy(
            nav.profile.map_types
        )

        nav.nav_dict = {
            'items': {},
        }

        grid = NavigatorMap(
            id='grid1',
            map_type='grid',
            serialem_item={},
            source_filepath='grid.mrc',
            filepath='grid.mrc',
        )

        lamella = NavigatorMap(
            id='lamella1',
            map_type='lamella',
            serialem_item={},
            source_filepath='lamella.mrc',
            filepath='lamella.mrc',
        )

        view = NavigatorMap(
            id='view1',
            map_type='view',
            serialem_item={},
            source_filepath='view.mrc',
            filepath='view.mrc',
        )

        tgt1 = NavigatorMap(
            id='tgt1',
            map_type='tgt',
            serialem_item={
                'MapFile': 'L01_tgt_001.mrc',
                'MapScaleMat': '1 0 0 1',
                'StageXYZ': '10 20 0',
            },
            source_filepath='tgt1.mrc',
            filepath='tgt1.mrc',
            resolution=np.array([
                0.001,
                0.001,
                0.001,
            ]),
        )

        tgt2 = NavigatorMap(
            id='tgt2',
            map_type='tgt',
            serialem_item={
                'MapFile': 'L01_tgt_002.mrc',
                'MapScaleMat': '2 0 0 2',
                'StageXYZ': '30 40 0',
            },
            source_filepath='tgt2.mrc',
            filepath='tgt2.mrc',
            resolution=np.array([
                0.002,
                0.002,
                0.002,
            ]),
        )

        nav.maps.add_many([
            grid,
            lamella,
            view,
            tgt1,
            tgt2,
        ])

        for map_ in nav.maps:
            nav.hierarchy.add_node(
                map_.map_type,
                map_.id,
            )

        nav.hierarchy.add_relation(
            'grid',
            'grid1',
            'lamella',
            'lamella1',
        )

        nav.hierarchy.add_relation(
            'lamella',
            'lamella1',
            'view',
            'view1',
        )

        nav.hierarchy.add_relation(
            'view',
            'view1',
            'tgt',
            'tgt1',
        )

        nav.hierarchy.add_relation(
            'view',
            'view1',
            'tgt',
            'tgt2',
        )

        return nav

    def test_add_tomograms(self):
        print(
            'Testing Navigator: '
            'add_tomograms ...'
        )

        nav = self._create_nav()

        tgt1 = nav.get_map(
            'tgt',
            'tgt1',
        )

        tgt2 = nav.get_map(
            'tgt',
            'tgt2',
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            tomo1_filepath = (
                tmpdir
                / 'L01_ts_001.mrc'
            )

            tomo2_filepath = (
                tmpdir
                / 'L01_ts_002.mrc'
            )

            self._create_mrc(
                tomo1_filepath,
                shape=(5, 20, 30),
            )

            self._create_mrc(
                tomo2_filepath,
                shape=(7, 40, 50),
            )

            nav.add_tomograms(
                tmpdir
            )

            tomo1 = nav.get_map(
                'tomo',
                '0',
            )

            tomo2 = nav.get_map(
                'tomo',
                '1',
            )

            self.assertEqual(
                nav.get_parent(
                    'tomo',
                    '0',
                ).id,
                'tgt1',
            )

            self.assertEqual(
                nav.get_parent(
                    'tomo',
                    '1',
                ).id,
                'tgt2',
            )

            self.assertEqual(
                tomo1.get_serialem_value(
                    'MapScaleMat'
                ),
                '1 0 0 1',
            )

            self.assertEqual(
                tomo1.get_serialem_value(
                    'StageXYZ'
                ),
                '10 20 0',
            )

            self.assertTrue(
                np.allclose(
                    tomo1.resolution,
                    tgt1.resolution,
                )
            )

            self.assertTrue(
                np.allclose(
                    tomo2.resolution,
                    tgt2.resolution,
                )
            )

            self.assertEqual(
                tomo1.shape,
                (
                    5,
                    20,
                    30,
                ),
            )

            self.assertEqual(
                tomo2.shape,
                (
                    7,
                    40,
                    50,
                ),
            )

    def test_add_tomograms_hierarchy_order(self):
        print(
            'Testing Navigator: '
            'add_tomograms hierarchy order ...'
        )

        nav = self._create_nav()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            self._create_mrc(
                tmpdir / 'L01_ts_001.mrc',
                shape=(5, 20, 30),
            )

            self._create_mrc(
                tmpdir / 'L01_ts_002.mrc',
                shape=(7, 40, 50),
            )

            nav.add_tomograms(
                tmpdir
            )

            result = [
                (
                    map_.map_type,
                    map_.id,
                    idx_path,
                )
                for map_, idx_path in nav
            ]

        expected = [
            ('grid', 'grid1', [0]),
            ('lamella', 'lamella1', [0, 0]),
            ('view', 'view1', [0, 0, 0]),
            ('tgt', 'tgt1', [0, 0, 0, 0]),
            ('tomo', '0', [0, 0, 0, 0, 0]),
            ('tgt', 'tgt2', [0, 0, 0, 1]),
            ('tomo', '1', [0, 0, 0, 1, 0]),
        ]

        self.assertEqual(
            result,
            expected,
        )
        print(
            'Testing Navigator: '
            'add_tomograms hierarchy order ...'
        )

        nav = self._create_nav()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            self._create_mrc(
                tmpdir / 'L01_ts_001.mrc',
                shape=(5, 20, 30),
            )

            self._create_mrc(
                tmpdir / 'L01_ts_002.mrc',
                shape=(7, 40, 50),
            )

            nav.add_tomograms(
                tmpdir
            )

            result = [
                (
                    map_.map_type,
                    map_.id,
                    idx_path,
                )
                for map_, idx_path in nav
            ]

        expected = [
            ('grid', 'grid1', [0]),
            ('lamella', 'lamella1', [0, 0]),
            ('view', 'view1', [0, 0, 0]),
            ('tgt', 'tgt1', [0, 0, 0, 0]),
            ('tomo', '0', [0, 0, 0, 0, 0]),
            ('tgt', 'tgt2', [0, 0, 0, 1]),
            ('tomo', '1', [0, 0, 0, 1, 0]),
        ]

        self.assertEqual(
            result,
            expected,
        )
        

if __name__ == '__main__':
    unittest.main()

