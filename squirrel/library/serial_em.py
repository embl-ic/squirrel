import numpy as np
from pathlib import Path

# ______________________________________________________________________________________________________________________
# Functions of the original linkmaps workflow

def get_unique_key(base_key, existing_keys):
    """Generate a unique key by appending -1, -2, etc. if necessary."""
    if base_key not in existing_keys:
        return base_key
    i = 1
    while f"{base_key}-{i}" in existing_keys:
        i += 1
    return f"{base_key}-{i}"


def navigator_file_to_dict(filepath):

    import re

    data = {}
    items = {}
    current_item = None

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):  # Ignore empty lines and comments
                continue

            # Match key-value pairs like "version = 2.00"
            match = re.match(r"(\w+)\s*=\s*(.+)", line)
            if match:
                key, value = match.groups()

                # Convert numerical values if possible
                try:
                    value = float(value) if '.' in value else int(value)
                except ValueError:
                    pass

                if current_item is None:
                    data[key] = value
                else:
                    items[current_item][key] = value
                continue

            # Match item sections like "[Item = 7-A]" or "[Item = empty]"
            # match = re.match(r"\[Item\s*=\s*([\w-]+)\]", line)
            match = re.match(r"\[Item\s*=\s*(.*?)\]", line)
            if match:
                base_item = match.group(1)
                current_item = get_unique_key(base_item, items)
                items[current_item] = {}
                continue

    data['items'] = items
    return data


def extend_navigator_dict(filepath, nav_dict):

    def _get_base_key(key):
        return re.sub(r'^(.*)-(0|[1-9]\d*)$', r'\1', key)

    def _get_items_with_base_key(d, base_key):
        """Yield all (full_key, item) pairs in d that share the same base_key."""
        for k, v in d.items():
            if _get_base_key(k) == base_key:
                yield k, v

    import re
    from copy import deepcopy

    out_dict = deepcopy(nav_dict)

    extend_dict = navigator_file_to_dict(filepath)

    assert extend_dict['AdocVersion'] == nav_dict['AdocVersion'], "AdocVersions do not match! Not sure if this is bad, but I'll not allow it"

    all_base_keys = list(np.unique([_get_base_key(x) for x in nav_dict['items'].keys()]))

    for key, nav_item in extend_dict['items'].items():

        clean_key = _get_base_key(key)

        if clean_key in all_base_keys:
            # Compare MapID to ALL items with the same base_key
            existing_items = list(_get_items_with_base_key(nav_dict['items'], clean_key))
            same_mapid = any(item['MapID'] == nav_item['MapID'] for _, item in existing_items)

            if not same_mapid:
                # Make a unique key for the new item
                unique_key = get_unique_key(clean_key, out_dict['items'])
                out_dict['items'][unique_key] = nav_item
                print('Added new item with existing base key:')
                print(f'key = {key}, clean_key = {clean_key}, unique_key = {unique_key}')
            else:
                print('Item exists, not adding it:')
                print(f'key = {key}, clean_key = {clean_key}')
                # MapID matches an existing one → check MapFile if present
                if 'MapFile' in nav_item:
                    # Find the item with matching MapID and compare MapFile
                    for _, item in existing_items:
                        if item['MapID'] == nav_item['MapID']:
                            assert item['MapFile'] == nav_item['MapFile']
                            break

        else:
            print('Added new item:')
            print(f'key = {key}, clean_key = {clean_key}')
            out_dict['items'][clean_key] = nav_item
            if not clean_key in all_base_keys:
                all_base_keys.append(clean_key)

    return out_dict


def get_map_items_by_map_file(nav_dict, endswith='_search.mrc'):

    out_items = {}

    for key, val in nav_dict['items'].items():
        if 'MapFile' in val:
            if val['MapFile'].endswith(endswith):
                out_items[key] = val

    return out_items


def get_resolution_from_mdoc(filepath, unit='micrometer'):

    import re

    with open(filepath, 'r') as file:
        mdoc_content = file.read()
    pixel_spacing_match = re.search(r'PixelSpacing\s*=\s*([\d.]+)', mdoc_content)
    pixel_spacing = float(pixel_spacing_match.group(1))

    if unit == 'micrometer':
        return pixel_spacing / 10000
    if unit == 'nanometer':
        return pixel_spacing / 10
    if unit == 'angstrom':
        return pixel_spacing
    raise ValueError(f'Invalid unit: {unit}')


def get_resolution_from_mrc_header(filepath, unit='micrometer'):
    import mrcfile

    with mrcfile.open(filepath, permissive=True) as mrc:
        pixel_spacing = mrc.voxel_size
    pixel_spacing = np.array([pixel_spacing.x, pixel_spacing.y])
    if not np.isclose(pixel_spacing[0], pixel_spacing[1]):
        raise ValueError(f'Assuming isotropic pixel spacing! Found: {pixel_spacing}')
    pixel_spacing = pixel_spacing[0]

    if unit == 'micrometer':
        return pixel_spacing / 10000
    if unit == 'nanometer':
        return pixel_spacing / 10
    if unit == 'angstrom':
        return pixel_spacing

    return pixel_spacing


def get_resolution_of_nav_item(nav_item, mdoc_dirpath=None, unit='micrometer', map_type=None):

    map_filepath = Path(nav_item['MapFile'].replace("\\", '/'))
    if mdoc_dirpath is None:
        mdoc_filepath = map_filepath.parent / f'{map_filepath.name}.mdoc'
    else:
        mdoc_filepath = Path(mdoc_dirpath) / f'{map_filepath.name}.mdoc'

    return get_resolution_from_mdoc(mdoc_filepath, unit=unit)


def get_raw_stage_xy_from_item(item_dict):

    if 'RawStageXY' in item_dict:
        return [float(x) for x in item_dict['RawStageXY'].split(' ')]  # [::-1]


def get_value_list_from_item(item_dict, name):
    if name in item_dict:
        return [float(x) for x in item_dict[name].split(' ')]


def get_value_from_item(item_dict, name):
    if name in item_dict:
        return float(item_dict[name])


def stage_to_image_coords(stage_coords, scale_mat):
    """
    Convert stage coordinates (microns) to pixel coordinates on the map image.

    Parameters:
        stage_coords (list of tuples): Stage coordinates (microns).
        scale_mat (list): [a, b, c, d] from MapScaleMat.

    Returns:
        list of tuples: Image pixel coordinates.
    """
    a, b, c, d = scale_mat
    affine = np.array([[a, b],
                       [c, d]])

    stage_coords = np.asarray(stage_coords)
    # FIXME: I'm still not sure if I need to do affine or affine.T
    #  so far, it didn't do much of a difference, because the matrices were quite symmetric
    img_coords = stage_coords @ affine.T
    return [tuple(coord) for coord in img_coords]


def get_search_map_on_grid_info(nav_filepath, grid_map_img_bin=8, search_map_img_bin=4, verbose=False):

    nav_filepath = Path(nav_filepath)

    nav_dict = navigator_file_to_dict(nav_filepath)

    # Get relevant navigator items
    search_map_items = get_map_items_by_map_file(nav_dict, '_search.mrc')
    grid_map_items = get_map_items_by_map_file(nav_dict, 'gridmap.st')

    # Get resolution of grid map
    grid_map_resolution = get_resolution_of_nav_item(
        grid_map_items[next(iter(grid_map_items))],
        nav_filepath.parent,
        unit='micrometer'
    )
    # Get the resolutions of the search maps (only to return it)
    search_map_resolutions = [
        get_resolution_of_nav_item(
            search_map_item,
            nav_filepath.parent,
            unit='micrometer'
        )
        for _, search_map_item in search_map_items.items()
    ]

    # Get the search map information
    search_map_names = []
    search_map_stage_positions = []
    search_map_item_list = []
    search_map_scale_mats = []
    for k, v in search_map_items.items():
        search_map_names.append(str(v['MapSection'] + 1))
        search_map_stage_positions.append(get_raw_stage_xy_from_item(v))
        search_map_item_list.append(v)
        search_map_scale_mats.append(np.array(get_value_list_from_item(v, 'MapScaleMat')) / search_map_img_bin)

    search_map_stage_positions = np.array(search_map_stage_positions)
    stage_xy = np.array(get_value_list_from_item(grid_map_items[next(iter(grid_map_items))], 'StageXYZ'))[:2]

    # # Use the affine transform to get the transformed stage positions
    grid_map_binning = get_value_from_item(grid_map_items[next(iter(grid_map_items))], 'MapBinning')
    map_scale_mat = np.array(
        get_value_list_from_item(grid_map_items[next(iter(grid_map_items))], 'MapScaleMat')
    ) / grid_map_img_bin

    transformed_stage_positions = np.array(stage_to_image_coords(
        (np.array(search_map_stage_positions) - np.array(stage_xy)),
        map_scale_mat
    ))

    return_dict = dict(
        search_map_names=search_map_names,
        search_map_item_list=search_map_item_list,
        search_map_positions=transformed_stage_positions,
        search_map_resolutions=search_map_resolutions,
        search_map_scale_mats=search_map_scale_mats,
        search_map_stage_xys=search_map_stage_positions,
        grid_map_resolution=grid_map_resolution,
        grid_map_scale_mat=map_scale_mat,
        grid_map_stage_xy=stage_xy,
        grid_map_binning=grid_map_binning
    )

    if verbose:
        print(f'return_dict = {return_dict}')

    return return_dict


def get_gridmap_filepath(nav_filepath, extension='mrc'):
    return list(Path(nav_filepath).parent.glob(f'gridmap_*.{extension}'))[0]


def get_nav_item_id_from_note(in_item):
    return in_item['Note'].split(' ')[0]


def get_view_map_items_by_drawn_id(view_map_items, drawn_id, nav_dict_items, return_dict=False):
    if return_dict:
        return {k: v for k, v in view_map_items.items() if
                nav_dict_items[get_nav_item_id_from_note(v)]['DrawnID'] == drawn_id}

    return_items = []
    for k, v in view_map_items.items():
        nav_item = nav_dict_items[get_nav_item_id_from_note(v)]
        if 'DrawnID' not in nav_item:
            print(f'Warning: "DrawnID" entry not in navigator item {get_nav_item_id_from_note(v)}')
            continue
        if nav_dict_items[get_nav_item_id_from_note(v)]['DrawnID'] == drawn_id:
            return_items.append(k)
    return return_items


def assign_view_maps_to_search_map(view_map_items, search_map_items, nav_dict_items, return_items=False):

    map_ids = {k: v['MapID'] for k, v in search_map_items.items()}

    if return_items:

        return {
            k: get_view_map_items_by_drawn_id(view_map_items, v, nav_dict_items, return_dict=True)
            for k, v in map_ids.items()
        }

    return {k: get_view_map_items_by_drawn_id(view_map_items, v, nav_dict_items) for k, v in map_ids.items()}


def get_view_on_search_map_info(
        nav_filepath,
        search_map_item,
        view_map_items,
        search_map_img_bin=4,
        view_map_img_bin=4,
        verbose=False
):

    nav_filepath = Path(nav_filepath)

    # Get resolution of search maps
    from squirrel.library.serial_em import get_resolution_of_nav_item
    search_map_resolution = get_resolution_of_nav_item(
        search_map_item,
        nav_filepath.parent,
        unit='micrometer'
    )
    # Get the resolutions of the view maps (only to return it)
    view_map_resolutions = [
        get_resolution_of_nav_item(
            view_map_item,
            nav_filepath.parent,
            unit='micrometer'
        )
        for _, view_map_item in view_map_items.items()
    ]

    # Get the view map information
    from squirrel.library.serial_em import get_raw_stage_xy_from_item
    view_map_names = []
    view_map_stage_positions = []
    view_map_item_list = []
    view_map_scale_mats = []
    view_map_binnings = []
    for k, v in view_map_items.items():
        view_map_names.append(str(v['MapSection'] + 1))
        view_map_stage_positions.append(get_raw_stage_xy_from_item(v))
        view_map_item_list.append(v)
        view_map_scale_mats.append(np.array(get_value_list_from_item(v, 'MapScaleMat')) / view_map_img_bin)
        view_map_binnings.append(get_value_from_item(v, 'MapBinning'))

    search_map_binning = get_value_from_item(search_map_item, 'MapBinning')
    stage_xy = np.array(get_value_list_from_item(search_map_item, 'StageXYZ'))[:2]
    map_scale_mat = np.array(get_value_list_from_item(search_map_item, 'MapScaleMat')) / search_map_img_bin

    transformed_stage_positions = np.array(stage_to_image_coords(
        (np.array(view_map_stage_positions) - np.array(stage_xy)),
        map_scale_mat
    ))

    return_dict = dict(
        view_map_names=view_map_names,
        view_map_item_list=view_map_item_list,
        view_map_positions=transformed_stage_positions,
        view_map_resolutions=view_map_resolutions,
        view_map_scale_mats=view_map_scale_mats,
        view_map_stage_xys=view_map_stage_positions,
        view_map_binnings=view_map_binnings,
        search_map_resolution=search_map_resolution,
        search_map_scale_mat=map_scale_mat,
        search_map_stage_xy=stage_xy,
        search_map_binning=search_map_binning
    )

    if verbose:
        print(f'return_dict = {return_dict}')

    return return_dict


def get_searchmap_filepath(search_map_item, nav_filepath, binning=4, pad_zeros=0, extension='mrc'):
    map_filepath = Path(search_map_item['MapFile'].replace("\\", '/'))
    map_section = int(search_map_item['MapSection']) + 1
    map_section_str = ('{:0' + str(pad_zeros) + 'd}').format(map_section)
    return Path(nav_filepath).parent / f'{map_filepath.stem}_{map_section_str}_bin{binning}.{extension}'

# ______________________________________________________________________________________________________________________


def get_filepath_from_nav_item(nav_filepath, map_item, item_id):
    map_filepath = Path(map_item[item_id].replace("\\", '/'))
    return Path(nav_filepath).parent / map_filepath.name


def get_map_filepath_from_nav_item(nav_filepath, map_item):
    return get_filepath_from_nav_item(nav_filepath, map_item, 'MapFile')


def get_map_items_by_glob(nav_dict, nav_filepath, glob='*.mrc'):
    from fnmatch import fnmatch

    out_items = {}
    test_fp = Path(nav_filepath).parent / glob

    for key, val in nav_dict['items'].items():
        if 'MapFile' in val:
            item_fp = get_map_filepath_from_nav_item(nav_filepath, val)

            if fnmatch(item_fp, test_fp):
                out_items[key] = val

    return out_items


def get_mdoc_filepath(mrc_filepath, check_exist=False):
    mrc_filepath = Path(mrc_filepath)
    mdoc_filepath = mrc_filepath.parent / f'{mrc_filepath.name}.mdoc'
    if mdoc_filepath.exists() or not check_exist:
        return mdoc_filepath
    raise FileNotFoundError(f'Mdoc filepath does not exist: {mdoc_filepath}')


def get_map_scale_xy(map_item):
    return get_value_list_from_item(map_item, 'StageXYZ')[:2]


def get_map_scale_matrix_from_item(item_dict):
    if 'MapScaleMat' in item_dict:
        mat = [float(x) for x in item_dict['MapScaleMat'].split(' ')]
        mat = np.array(mat).reshape([2, 2], order='C')
        return mat
    return None


def get_map_shape(map_item, binning):
    bin_factor = get_value_from_item(map_item, 'MapBinning') / (
                get_value_from_item(map_item, 'MontBinning') * binning)
    return (np.array(get_value_list_from_item(map_item, 'MapWidthHeight')) * bin_factor).astype(int)


def get_contrast_limits_from_map(
        map_fp
):
    from squirrel.library.data import get_contrast_limits
    import mrcfile
    with mrcfile.open(map_fp, permissive=True) as mrc:
        img = mrc.data
    return get_contrast_limits(img)


class Navigator:

    def __init__(
            self,
            filepaths,
            map_types: list[str] = None,
            map_hierarchy: list[str] = None,
            search_strings: dict[str, str] = None,
            map_binnings: dict[str, int] = None,
            skip_key_hierarchy_init: bool = False,
            verbose: bool = False
    ):

        self.verbose = verbose
        self.map_types = map_types if map_types is not None else ['grid']
        self.search_strings = search_strings if search_strings is not None else dict(grid='gridmap.st')
        self.map_binnings = map_binnings if map_binnings is not None else dict(zip(self.map_types, [1] * len(self.map_types)))

        assert len(self.search_strings) == len(self.map_types), 'Number of search strings must match number of map types'
        assert len(self.map_binnings) == len(self.map_types), 'Number of map binnings must match number of map types'

        self.filepath, self.all_filepaths, self.nav_dict = self._nav_filepaths_to_dict(filepaths)

        # It is important to add the items like this because some items might depend on the earlier added ones being
        #   present already
        self.map_items_dict = dict()
        for map_type in self.map_types:
            self.map_items_dict[map_type] = self._get_map_items_dict(map_type)

        self.map_ids = {map_type: sorted(self.map_items_dict[map_type].keys()) for map_type in map_types}

        self.map_filepaths = {map_type: self._get_map_filepaths(map_type) for map_type in self.map_types}

        self.mdoc_filepaths = {map_type: self._get_mdoc_filepaths(map_type) for map_type in self.map_types}

        self.map_resolutions = {map_type: self._get_map_resolutions(map_type) for map_type in self.map_types}

        self.map_shapes = {map_type: self._get_map_shapes(map_type) for map_type in self.map_types}

        self.map_contrast_limits = {map_type: None for map_type in self.map_types}

        self.map_sec_ids = {map_type: self._get_map_sec_ids(map_type) for map_type in self.map_types}

        self.map_hierarchy = map_hierarchy.copy()
        if not skip_key_hierarchy_init:
            self._build_key_hierarchy()

        self._count = [0] * len(self.map_types)

    @staticmethod
    def _nav_filepaths_to_dict(nav_filepaths):
        if type(nav_filepaths) == str:
            return Path(nav_filepaths), [], navigator_file_to_dict(nav_filepaths)

        all_filepaths = [Path(fp) for fp in nav_filepaths]
        filepath = all_filepaths[0]

        nav_dict = navigator_file_to_dict(filepath)
        for fp in all_filepaths[1:]:
            nav_dict = extend_navigator_dict(fp, nav_dict)

        return filepath, all_filepaths, nav_dict

    @staticmethod
    def _sort_key(item):
        import re
        match = re.match(r"(\d+)(.*)", item)
        if match:
            number = int(match.group(1))
            suffix = match.group(2)
            return number, suffix
        return float('inf'), item

    def _function_on_property(self, map_type, prop_name, func, **kwargs):
        try:
            return {k: func(map_type=map_type, key=k, item=v, **kwargs) for k, v in getattr(self, prop_name)[map_type].items()}
        except TypeError as original_error:
            try:
                return {k: func(v, **kwargs) for k, v in getattr(self, prop_name)[map_type].items()}
            except Exception as e:
                print('------------------------')
                print(f'Error: {e}')
                print('------------------------')
                raise original_error

    def _get_map_items_dict(self, map_type):
        return get_map_items_by_glob(self.nav_dict, self.filepath, self.search_strings[map_type])

    def _get_map_filepath(self, item):
        return get_map_filepath_from_nav_item(self.filepath, item)

    def _get_map_filepaths(self, map_type):
        # print(map_type)
        return self._function_on_property(map_type, 'map_items_dict', self._get_map_filepath)

    def _get_mdoc_filepaths(self, map_type):
        return self._function_on_property(map_type, 'map_filepaths', get_mdoc_filepath)

    def _get_map_resolution(self, map_type, key, item=None):
        # try:
        #     return get_resolution_from_mdoc(self.mdoc_filepaths[map_type][key], unit='micrometer')
        # except (FileNotFoundError, KeyError):
        return get_resolution_from_mrc_header(self.map_filepaths[map_type][key], unit='micrometer')

    def _get_map_resolutions(self, map_type):
        return self._function_on_property(map_type, 'map_items_dict', self._get_map_resolution)

    def _get_map_shape(self, map_type, key, item=None):
        import mrcfile
        with mrcfile.open(item, permissive=True, header_only=True) as mrc:
            shape = mrc.header.nz, mrc.header.ny, mrc.header.nx
        return np.array(shape)

    def _get_map_shapes(self, map_type):
        return self._function_on_property(map_type, 'map_filepaths', self._get_map_shape)

    def _get_affine(
            self, map_type, key, item, binning,
            invert=False,
            full_square=False,
            flatten=False,
            rotate_90=False,
            is_3d=False
    ):

        xy = get_map_scale_xy(item)  # Gets StageXYZ entry from SerialEM
        mat = get_map_scale_matrix_from_item(item)  # gets MapScaleMatrix entry from SerialEM
        from squirrel.library.io import get_mrc_shape
        img_shp = get_mrc_shape(self.map_filepaths[map_type][key])[::-1]
        if not is_3d:
            img_shp = img_shp[:2]
        map_resolution = self.map_resolutions[map_type][key]

        def scale_to_one(x):
            return 2 ** round(-np.log2(x))

        bin_factor = scale_to_one(np.abs(mat[:2].max() * map_resolution)) * map_resolution
        bin_factor_matrix = np.eye(4)
        bin_factor_matrix[:2, :2] = bin_factor_matrix[:2, :2] * bin_factor

        affine = np.array([
            [mat[0, 0], mat[0, 1], 0, img_shp[0] / 2 * (map_resolution)],  # * binning)],
            [mat[1, 0], mat[1, 1], 0, img_shp[1] / 2 * (map_resolution)],  # * binning)],
            [0, 0, 1, 0 if not is_3d else img_shp[2] / 2 * (map_resolution)],  # * binning)],
            [0, 0, 0, 1]
        ])
        affine = affine @ bin_factor_matrix

        if rotate_90:
            R = np.array([[0, -1], [1,  0]])
            affine[:2, :2] = affine[:2, :2] @ R

        if invert:
            affine = np.linalg.inv(affine)
        if not full_square:
            affine = affine[:3]
        affine[0, 3] += xy[0]
        affine[1, 3] += xy[1]
        if flatten:
            affine = affine.flatten()
        return affine

    def _get_map_full_affine(
            self, map_type, key, item, binning=1,
            apply_affine=None, invert=False, full_square=False, flatten=False, is_3d=False, rotate_90=False
    ):

        if apply_affine is not None:
            affine = self._get_affine(map_type, key, item, binning, invert=invert, full_square=full_square, flatten=False, is_3d=is_3d, rotate_90=rotate_90)
            if flatten:
                return (apply_affine @ affine).flatten()
            return apply_affine @ affine

        return self._get_affine(map_type, key, item, binning, invert=invert, full_square=full_square, flatten=flatten, is_3d=is_3d, rotate_90=rotate_90)

    def get_grid_key(self):
        return next(iter(self.map_items_dict['grid']))

    def get_grid_item(self):
        key = self.get_grid_key()
        return self.map_items_dict['grid'][key]

    def get_map_full_affines(self, map_type, stage_coordinate_system=False, is_3d=False, rotate_90=False):

        kwargs = dict(
            apply_affine=(
                None if stage_coordinate_system else
                self._get_map_full_affine('grid', self.get_grid_key(), self.get_grid_item(), self.map_binnings['grid'], invert=False, full_square=True, flatten=False)
            ),
            binning=self.map_binnings[map_type],
            invert=True, full_square=True, flatten=True, is_3d=is_3d, rotate_90=rotate_90
        )
        return self._function_on_property(map_type, 'map_items_dict', self._get_map_full_affine, **kwargs)

    @staticmethod
    def _get_map_sec_id(map_item):
        import re
        match = re.search(r'Sec (\d+)', map_item['Note'])
        return int(match.group(1)) if match else None

    def _get_map_sec_ids(self, map_type):
        return self._function_on_property(map_type, 'map_items_dict', self._get_map_sec_id)

    def get_contrast_limits(self, map_type):
        if self.map_contrast_limits[map_type] is None:
            print(f'Computing contrast limits for {map_type}')
            self.map_contrast_limits[map_type] = self._function_on_property(map_type, 'map_filepaths', get_contrast_limits_from_map)
        return self.map_contrast_limits[map_type]

    def _get_key_dependencies_for_map_type(self, map_type: str) -> list:
        if self.map_hierarchy is None:
            return list(self.map_items_dict[map_type].keys())
        raise NotImplementedError('map_hierarchy != None is not implemented for the base class! '
                                  'Classes that need this must implement the functionality '
                                  'and thus override this function.')

    def _build_key_hierarchy(self):
        self.key_hierarchy = dict()
        ordered_map_types = self.map_hierarchy if self.map_hierarchy is not None else self.map_types
        for map_type in ordered_map_types:
            self.key_hierarchy[map_type] = self._get_key_dependencies_for_map_type(map_type)

    def _key_generator(self):

        def loop_func(hierarchy, this_key_hierarchy, idx_path = None):
            this_hierarchy = hierarchy.pop(0)

            for idx, item in enumerate(this_key_hierarchy):
                this_idx_path = [idx] if idx_path is None else idx_path + [idx]
                yield this_hierarchy, item, this_idx_path

                if hierarchy:

                    this_child_key_hierarchy = self.key_hierarchy[hierarchy[0]]
                    try:
                        for jdx in this_idx_path:
                            this_child_key_hierarchy = this_child_key_hierarchy[jdx]
                    except IndexError:
                        continue
                    yield from loop_func(hierarchy.copy(), this_child_key_hierarchy, this_idx_path)

        yield from loop_func(self.map_hierarchy.copy(), self.key_hierarchy['grid'])

    def __iter__(self):
        return self._key_generator()

    def get_parent_map_type(self, map_type):
        if not map_type in self.map_hierarchy:
            raise ValueError(f'Invalid map_type = {map_type}')
        idx = self.map_hierarchy.index(map_type)
        if idx == 0:
            return None
        return self.map_hierarchy[idx - 1]

    def get_child_map_type(self, map_type):
        if not map_type in self.map_hierarchy:
            raise ValueError(f'Invalid map_type = {map_type}')
        idx = self.map_hierarchy.index(map_type)
        if idx == len(self.map_hierarchy):
            return None
        return self.map_hierarchy[idx + 1]


class SingleParticleNavigator(Navigator):

    MAP_TYPES = ['grid', 'search', 'view', 'record']

    DEFAULT_SEARCH_STRINGS = dict(
        grid='gridmap.st',
        search='*_search.mrc',
        view='*_view.mrc',
        record='*_record.mrc',
    )

    DEFAULT_MAP_BINNINGS = dict(
        grid=8,
        search=4,
        view=1,
        record=1
    )

    DEFAULT_MATCH_TO_PARENTS = dict(
        view=dict(  # How to match views with lamellae
            lamella=['MapFile', r'L_(\d{2})'],
            view=['MapFile', r'L(\d{2})']
        ),
        tgt=dict(  # How to match tgts with views
            view=['MapFile', r'L(\d{2})_tgt_(\d{3})'],
            tgt=['MapFile', r'L(\d{2})_tgt_(\d{3})']
        )
    )

    def __init__(
            self,
            filepaths: str|list[str],
            search_strings: dict|None = None,
            map_binnings: dict|None = None,
            map_types: list[str]|None = None,
            match_to_parents: dict|None = None,
            stitched_dirpath: str|None = None
    ):

        map_types = self.MAP_TYPES if map_types is None else map_types
        if search_strings is not None:
            map_types = list(search_strings.keys())
            if any([map_type not in self.MAP_TYPES for map_type in map_types]):
                raise ValueError(f'Invalid map type encountered! {map_types}')

        if search_strings is None:
            search_strings = {map_type: self.DEFAULT_SEARCH_STRINGS[map_type] for map_type in map_types}
        if map_binnings is None:
            map_binnings = {map_type: self.DEFAULT_MAP_BINNINGS[map_type] for map_type in map_types}
        self.match_to_parents = match_to_parents
        if self.match_to_parents is None:
            self.match_to_parents = {map_type: self.DEFAULT_MATCH_TO_PARENTS[map_type] for map_type in map_types if map_type in self.DEFAULT_MATCH_TO_PARENTS}
        self.stitched_dirpath = Path(stitched_dirpath) if stitched_dirpath is not None else None

        super().__init__(
            filepaths,
            map_types=map_types,
            map_hierarchy=map_types,
            search_strings=search_strings,
            map_binnings=map_binnings,
            skip_key_hierarchy_init=True
        )

        self.match_ids = {map_type: self._get_map_match_ids(map_type) for map_type in map_types}

        self._build_match_dicts()
        self._build_key_hierarchy()

    # TODO: Put into general Navigator class
    def _build_match_dicts_for_map(self, map_type):
        parent_ids, child_ids = self._assign_maps_to_maps(map_type, self.get_parent_map_type(map_type), self.match_ids[map_type])
        self.match_dict_fwd[self.get_parent_map_type(map_type)] = dict(zip(parent_ids, child_ids))
        self.match_dict_bkw[map_type] = dict()
        for idx, siblings in enumerate(child_ids):
            for child in siblings:
                self.match_dict_bkw[map_type][child] = parent_ids[idx]

    # TODO: Put general form into general Navigator class
    def _build_match_dicts(self):
        self.match_dict_bkw = dict()
        self.match_dict_fwd = dict()

        for k, v in self.match_ids.items():
            if k == 'grid':
                self.match_dict_bkw[k] = None
            if k == 'search':
                self.match_dict_fwd[self.get_parent_map_type(k)] = {self.map_ids[self.get_parent_map_type(k)][0]: self.map_ids[k]}
                self.match_dict_bkw[k] = {k: self.map_ids['grid'][0] for k in self.map_ids[k]}
            if k in ['view', 'record']:
                self._build_match_dicts_for_map(k)

    def _get_map_items_dict(self, map_type):

        if map_type == 'grid' and self.search_strings['grid'] is None:
            key = next(iter(self.nav_dict['items']))
            map_items = {key: self.nav_dict['items'][key]}
            if not map_items[key]['MapFile'].endswith('.mrc') and not map_items[key]['MapFile'].endswith('.st'):
                raise ValueError(f'Map items need to end with "*.mrc" or "*.st". Found "{map_items[key]["MapFile"]}"')
            return map_items

        if map_type == 'grid' and self.search_strings['grid'] is not None:
            # Here we have to make sure to really only allow one item!
            map_items = get_map_items_by_glob(self.nav_dict, self.filepath, self.search_strings[map_type])
            if len(map_items) == 0:
                raise ValueError('No grid map item was found!')
            if len(map_items) > 1:
                # We now have to check if we can exclude all except one
                valid_map_items = {}
                for k, v in map_items.items():
                    fp = get_map_filepath_from_nav_item(self.filepath, v)
                    grid_map_filepath = self._get_grid_map_filepath(fp, v, allow_not_exist=True)
                    if grid_map_filepath is not None:
                        valid_map_items[k] = v
                if len(valid_map_items) == 1:
                    return valid_map_items
                # Seemingly we cannot exclude enough or all of them are invalid
                raise ValueError(f'{len(valid_map_items)} valid grid map items were found, however exactly one valid grid map item is allowed!')
            return map_items

        return super()._get_map_items_dict(map_type)

    # TODO: Put into general Navigator class
    def _assign_maps_to_maps(self, map_type, ref_map_type, ref_map_ids):
        ref_map = ref_map_ids[ref_map_type]
        this_map = ref_map_ids[map_type]

        # Step 1: sort keys of `ref_map` by their values
        sorted_keys = sorted(list(ref_map.keys()))

        # Step 2: group keys of `this_map` by their values
        grouped_this_map = {}
        for key, val in this_map.items():
            grouped_this_map.setdefault(val, []).append(key)

        # Step 3: order grouped lists according to sorted `ref_map`
        sorted_groups_this_map = [sorted(grouped_this_map.get(ref_map[k], [])) for k in sorted_keys]

        return sorted_keys, sorted_groups_this_map

    # TODO: This is a very generic function. Put elsewhere. For now it could go outside of the class but stay in this py file.
    @staticmethod
    def _recursive_replace(recursive_list, mapping):
        def recurse(obj):
            if isinstance(obj, list):
                return [recurse(item) for item in obj]
            else:
                return mapping.get(obj, obj)  # fallback if not found

        return recurse(recursive_list)

    def _get_key_dependencies_for_map_type(self, map_type):

        if map_type == 'grid':
            return self.map_ids[map_type]

        if map_type in ['search', 'view', 'record']:
            # We are assuming only one grid, so all lamella are dependent on that one
            return self._recursive_replace(self.key_hierarchy[self.get_parent_map_type(map_type)], self.match_dict_fwd[self.get_parent_map_type(map_type)])

        raise ValueError(f'Invalid map_type: {map_type}')

    # TODO: Put into general Navigator class
    @staticmethod
    def _match_regex(s, regex):
        import re
        match = re.search(regex, str(s))
        if match:
            return ''.join(match.groups()) or match.group(0)

    # TODO: Put into general Navigator class
    def _get_map_map_id(self, map_type, key, item, match_description):

        if len(match_description) == 2:
            item_key, regex = match_description
            s = item[item_key]
            map_id = self._match_regex(s, regex)

        if len(match_description) == 4:
            item_key, regex = match_description[:2]
            s = item[item_key]
            map_id = self._match_regex(s, regex)
            item_key, regex = match_description[2:]
            s = self.nav_dict['items'][map_id][item_key]
            map_id = self._match_regex(s, regex)

        if len(match_description) not in [2, 4]:
            raise ValueError('Invalid match description: Supply either two or four values in the format ["key", "regex"] or ["key1", "regex1", "key2", "regex2"]')

        return map_id

    # TODO: Put into general Navigator class
    def _get_map_map_ids(self, map_type, match_description):
        return self._function_on_property(map_type, 'map_items_dict', self._get_map_map_id, match_description=match_description)

    # TODO: Put into general Navigator class
    def _get_map_match_ids(self, map_type):

        if map_type in self.match_to_parents:
            matches = self.match_to_parents[map_type]
            return {k: self._get_map_map_ids(k, v) for k, v in matches.items()}

        return None

    def _get_grid_map_filepath(self, fp, item, allow_not_exist=False):
        binning = self.map_binnings['grid']
        parent_fp = self.stitched_dirpath if self.stitched_dirpath is not None else fp.parent
        name = self._match_regex(item['Note'], r"^Grid \d{2} (\S+)")
        grid_map_fp = parent_fp / f'{fp.stem}_{name}_bin{binning}{fp.suffix}'

        if grid_map_fp.exists():
            return grid_map_fp

        if fp.suffix.lower() != '.mrc':
            grid_map_fp_mrc = parent_fp / f'{fp.stem}_{name}_bin{binning}.mrc'
            if grid_map_fp_mrc.exists():
                return grid_map_fp_mrc

        if allow_not_exist:
            print(f'Warning: Grid map item not found: {fp}')
            print(f'         Unsuccessful with {grid_map_fp.name}{"/.mrc" if fp.suffix != ".mrc" else ""}')
            return None
        raise FileNotFoundError(f'Grid map file not found for "{fp}"\n'
                                f'    Unsuccessful with {grid_map_fp.name}{"/.mrc" if fp.suffix != ".mrc" else ""}')

    def _get_search_map_filepath(self, fp, item):
        import re
        binning = self.map_binnings['search']
        parent_fp = self.stitched_dirpath if self.stitched_dirpath is not None else fp.parent
        section_id = int(self._match_regex(item['Note'], r"^Sec (\d+) ")) + 1

        pattern = rf"{fp.stem}_0*{section_id}_bin{binning}{fp.suffix}"

        for candidate_fp in parent_fp.iterdir():
            if re.search(pattern, candidate_fp.name):
                return candidate_fp

        if fp.suffix.lower() != '.mrc':

            pattern_mrc = rf"{fp.stem}_0*{section_id}_bin{binning}.mrc"

            for candidate_fp in parent_fp.iterdir():
                if re.search(pattern_mrc, candidate_fp.name):
                    return candidate_fp

        raise FileNotFoundError(f'Search map file not found for "{fp}"\n'
                                f'    Unsuccessful with {pattern}{"/.mrc" if fp.suffix != ".mrc" else ""}')

    def _get_view_map_filepath(self, fp, item):
        import re
        if not self.map_binnings['view'] == 1:
            raise NotImplementedError('View map only implemented for full resolution!')

        parent_fp = self.stitched_dirpath if self.stitched_dirpath is not None else fp.parent
        section_id = int(self._match_regex(item['Note'], r"Sec (\d+) ")) + 1

        pattern = rf"{fp.stem}_0*{section_id}{fp.suffix}"

        for candidate_fp in parent_fp.iterdir():
            if re.search(pattern, candidate_fp.name):
                return candidate_fp

        if fp.suffix.lower() != '.mrc':

            pattern_mrc = rf"{fp.stem}_0*{section_id}.mrc"

            for candidate_fp in parent_fp.iterdir():
                if re.search(pattern_mrc, candidate_fp.name):
                    return candidate_fp

        raise FileNotFoundError(f'View map file not found for "{fp}"\n'
                                f'    Unsuccessful with {pattern}{"/.mrc" if fp.suffix != ".mrc" else ""}')

    def _get_record_map_filepath(self, fp, item):
        import re
        if not self.map_binnings['record'] == 1:
            raise NotImplementedError('Record map only implemented for full resolution!')

        parent_fp = self.stitched_dirpath if self.stitched_dirpath is not None else fp.parent
        section_id = int(self._match_regex(item['Note'], r"Sec (\d+) ")) + 1

        pattern = rf"{fp.stem}_0*{section_id}{fp.suffix}"

        for candidate_fp in parent_fp.iterdir():
            if re.search(pattern, candidate_fp.name):
                return candidate_fp

        if fp.suffix.lower() != '.mrc':

            pattern_mrc = rf"{fp.stem}_0*{section_id}.mrc"

            for candidate_fp in parent_fp.iterdir():
                if re.search(pattern_mrc, candidate_fp.name):
                    return candidate_fp

        raise FileNotFoundError(f'View map file not found for "{fp}"\n'
                                f'    Unsuccessful with {pattern}{"/.mrc" if fp.suffix != ".mrc" else ""}')

    def _get_map_filepath(self, map_type, key, item):
        fp = super()._get_map_filepath(item)
        print(fp)
        if map_type == 'grid':
            return self._get_grid_map_filepath(fp, item)
        if map_type == 'search':
            return self._get_search_map_filepath(fp, item)
        if map_type == 'view':
            return self._get_view_map_filepath(fp, item)
        if map_type == 'record':
            return self._get_record_map_filepath(fp, item)
        return fp

    # def _get_grid_mdoc_filepath(self, item):
    #     fp = get_map_filepath_from_nav_item(self.filepath, item)
    #     return get_mdoc_filepath(fp)

    # def _get_lamella_mdoc_filepath(self, item):
    #     fp = get_map_filepath_from_nav_item(self.filepath, item)
    #     return get_mdoc_filepath(fp)

    # def _get_mdoc_filepaths(self, map_type):
    #     if map_type == 'grid':
    #         return self._function_on_property(map_type, 'map_items_dict', self._get_grid_mdoc_filepath)
    #     if map_type == 'lamella':
    #         return self._function_on_property(map_type, 'map_items_dict', self._get_lamella_mdoc_filepath)
    #     return super()._get_mdoc_filepaths(map_type)

    # def find_item(self, map_type, serial_em_item, regex, target_value):
    #     for map_id in self.map_ids[map_type]:
    #         s = self.map_items_dict[map_type][map_id][serial_em_item]
    #         candidate = self._match_regex(s, regex)
    #         if candidate == target_value:
    #             return map_id


class TomoCLEMNavigator(Navigator):

    MAP_TYPES = ['grid', 'lamella', 'view', 'tgt']

    DEFAULT_SEARCH_STRINGS = dict(
        grid=None,
        lamella='L_*.map',
        view='L??_tgt_???_view.mrc',
        tgt='L??_tgt_???.mrc'
    )

    DEFAULT_MAP_BINNINGS = dict(
        grid=8,
        lamella=8,
        view=1,
        tgt=1
    )

    DEFAULT_MATCH_TO_PARENTS = dict(
        view=dict(  # How to match views with lamellae
            lamella=['MapFile', r'L_(\d{2})'],
            view=['MapFile', r'L(\d{2})']
        ),
        tgt=dict(  # How to match tgts with views
            view=['MapFile', r'L(\d{2})_tgt_(\d{3})'],
            tgt=['MapFile', r'L(\d{2})_tgt_(\d{3})']
        )
    )

    def __init__(
            self,
            filepaths: str|list[str],
            search_strings: dict|None = None,
            map_binnings: dict|None = None,
            map_types: list[str]|None = None,
            match_to_parents: dict|None = None,
            stitched_dirpath: str|None = None
    ):

        map_types = self.MAP_TYPES if map_types is None else map_types
        if search_strings is not None:
            map_types = list(search_strings.keys())
            if any([map_type not in self.MAP_TYPES for map_type in map_types]):
                raise ValueError(f'Invalid map type encountered! {map_types}')

        if search_strings is None:
            search_strings = {map_type: self.DEFAULT_SEARCH_STRINGS[map_type] for map_type in map_types}
        if map_binnings is None:
            map_binnings = {map_type: self.DEFAULT_MAP_BINNINGS[map_type] for map_type in map_types}
        self.match_to_parents = match_to_parents
        if self.match_to_parents is None:
            self.match_to_parents = {map_type: self.DEFAULT_MATCH_TO_PARENTS[map_type] for map_type in map_types if map_type in self.DEFAULT_MATCH_TO_PARENTS}
        self.stitched_dirpath = Path(stitched_dirpath) if stitched_dirpath is not None else None

        super().__init__(
            filepaths,
            map_types=map_types,
            map_hierarchy=map_types,
            search_strings=search_strings,
            map_binnings=map_binnings,
            skip_key_hierarchy_init=True
        )

        self.match_ids = {map_type: self._get_map_match_ids(map_type) for map_type in map_types}

        self._build_match_dicts()
        self._build_key_hierarchy()

    def _build_match_dicts_for_map(self, map_type):
        parent_ids, child_ids = self._assign_maps_to_maps(map_type, self.get_parent_map_type(map_type), self.match_ids[map_type])
        self.match_dict_fwd[self.get_parent_map_type(map_type)] = dict(zip(parent_ids, child_ids))
        self.match_dict_bkw[map_type] = dict()
        for idx, siblings in enumerate(child_ids):
            for child in siblings:
                self.match_dict_bkw[map_type][child] = parent_ids[idx]

    def _build_match_dicts(self):
        self.match_dict_bkw = dict()
        self.match_dict_fwd = dict()

        for k, v in self.match_ids.items():
            if k == 'grid':
                self.match_dict_bkw[k] = None
            if k == 'lamella':
                self.match_dict_fwd[self.get_parent_map_type(k)] = {self.map_ids[self.get_parent_map_type(k)][0]: self.map_ids[k]}
                self.match_dict_bkw[k] = {k: self.map_ids['grid'][0] for k in self.map_ids[k]}
            if k in ['view', 'tgt']:
                self._build_match_dicts_for_map(k)

    def _get_map_items_dict(self, map_type):

        if map_type == 'grid' and self.search_strings['grid'] is None:
            key = next(iter(self.nav_dict['items']))
            map_items = {key: self.nav_dict['items'][key]}
            if not map_items[key]['MapFile'].endswith('.mrc') and not map_items[key]['MapFile'].endswith('.st'):
                raise ValueError(f'Map items need to end with "*.mrc" or "*.st". Found "{map_items[key]["MapFile"]}"')
            return map_items

        return super()._get_map_items_dict(map_type)

    def _assign_maps_to_maps(self, map_type, ref_map_type, ref_map_ids):
        ref_map = ref_map_ids[ref_map_type]
        this_map = ref_map_ids[map_type]

        # Step 1: sort keys of `ref_map` by their values
        sorted_keys = sorted(list(ref_map.keys()))

        # Step 2: group keys of `this_map` by their values
        grouped_this_map = {}
        for key, val in this_map.items():
            grouped_this_map.setdefault(val, []).append(key)

        # Step 3: order grouped lists according to sorted `ref_map`
        sorted_groups_this_map = [sorted(grouped_this_map.get(ref_map[k], [])) for k in sorted_keys]

        return sorted_keys, sorted_groups_this_map

    @staticmethod
    def _recursive_replace(recursive_list, mapping):
        def recurse(obj):
            if isinstance(obj, list):
                return [recurse(item) for item in obj]
            else:
                return mapping.get(obj, obj)  # fallback if not found

        return recurse(recursive_list)

    def _get_key_dependencies_for_map_type(self, map_type):

        if map_type == 'grid':
            return self.map_ids[map_type]

        if map_type in ['lamella', 'view', 'tgt', 'tomo']:
            # We are assuming only one grid, so all lamella are dependent on that one
            return self._recursive_replace(self.key_hierarchy[self.get_parent_map_type(map_type)], self.match_dict_fwd[self.get_parent_map_type(map_type)])

        raise ValueError(f'Invalid map_type: {map_type}')

    @staticmethod
    def _match_regex(s, regex):
        import re
        match = re.search(regex, s)
        if match:
            return ''.join(match.groups()) or match.group(0)

    def _get_map_map_id(self, map_type, key, item, match_description):

        item_key, regex = match_description
        s = item[item_key]
        map_id = self._match_regex(s, regex)

        return map_id

    def _get_map_map_ids(self, map_type, match_description):
        return self._function_on_property(map_type, 'map_items_dict', self._get_map_map_id, match_description=match_description)

    def _get_map_match_ids(self, map_type):

        if map_type in self.match_to_parents:
            matches = self.match_to_parents[map_type]
            return {k: self._get_map_map_ids(k, v) for k, v in matches.items()}

        return None

    def _get_grid_map_filepath(self, fp):
        binning = self.map_binnings['grid']
        parent_fp = self.stitched_dirpath if self.stitched_dirpath is not None else fp.parent
        grid_map_fp = parent_fp / f'{fp.stem}_stitched_grid01_bin{binning}{fp.suffix}'

        if grid_map_fp.exists():
            return grid_map_fp

        if fp.suffix.lower() != '.mrc':
            grid_map_fp_mrc = parent_fp / f'{fp.stem}_stitched_grid01_bin{binning}.mrc'
            if grid_map_fp_mrc.exists():
                return grid_map_fp_mrc

        raise FileNotFoundError(f'Grid map file not found for "{fp}"')

    def _get_lamella_map_filepath(self, fp):
        binning = self.map_binnings['lamella']
        parent_fp = self.stitched_dirpath if self.stitched_dirpath is not None else fp.parent
        lamella_map_fp = parent_fp / f'{fp.stem}_stitched_grid01_bin{binning}{fp.suffix}'

        if lamella_map_fp.exists():
            return lamella_map_fp

        if fp.suffix.lower() != '.mrc':
            lamella_map_fp = parent_fp / f'{fp.stem}_stitched_grid01_bin{binning}.mrc'
            if lamella_map_fp.exists():
                return lamella_map_fp

        raise FileNotFoundError(f'Lamella map file not found for "{fp}"')

    def _get_view_map_filepath(self, fp):
        # Check in the same directory
        view_map_fp = self.filepath.parent / fp.name
        if view_map_fp.exists():
            return view_map_fp
        # Try with 'pace'
        view_map_fp = self.filepath.parent.parent / 'pace' / fp.name
        if view_map_fp.exists():
            return view_map_fp
        # Fail
        raise FileNotFoundError(f'View map file not found for "{fp}"')

    def _get_tgt_map_filepath(self, fp):
        # Check in the same directory
        map_fp = self.filepath.parent / fp.name
        if map_fp.exists():
            return map_fp
        # Try with 'pace'
        map_fp = self.filepath.parent.parent / 'pace' / fp.name
        if map_fp.exists():
            return map_fp
        # Fail
        return self.filepath.parent.parent / 'pace' / fp.name

    def _get_map_filepath(self, map_type, key, item):
        fp = super()._get_map_filepath(item)
        print(fp)
        if map_type == 'grid':
            return self._get_grid_map_filepath(fp)
        if map_type == 'lamella':
            return self._get_lamella_map_filepath(fp)
        if map_type == 'view':
            return self._get_view_map_filepath(fp)
        if map_type == 'tgt':
            return self._get_tgt_map_filepath(fp)
        if map_type == 'ts':
            raise NotImplementedError
        return fp

    def _get_grid_mdoc_filepath(self, item):
        fp = get_map_filepath_from_nav_item(self.filepath, item)
        return get_mdoc_filepath(fp)

    def _get_lamella_mdoc_filepath(self, item):
        fp = get_map_filepath_from_nav_item(self.filepath, item)
        return get_mdoc_filepath(fp)

    def _get_mdoc_filepaths(self, map_type):
        if map_type == 'grid':
            return self._function_on_property(map_type, 'map_items_dict', self._get_grid_mdoc_filepath)
        if map_type == 'lamella':
            return self._function_on_property(map_type, 'map_items_dict', self._get_lamella_mdoc_filepath)
        return super()._get_mdoc_filepaths(map_type)

    def find_item(self, map_type, serial_em_item, regex, target_value):
        for map_id in self.map_ids[map_type]:
            s = self.map_items_dict[map_type][map_id][serial_em_item]
            candidate = self._match_regex(s, regex)
            if candidate == target_value:
                return map_id

    def add_tomograms(
            self,
            dirpath,
            pattern='*.mrc',
            match_parent=['MapFile', r'L(\d{2})_ts_(\d{3})'],
            match_self=r'L(\d{2})_ts_(\d{3})'
            # parent_map_type='tgt'
    ):

        self.map_types.append('tomo')
        self.map_hierarchy.append('tomo')
        self.map_binnings['tomo'] = 1
        self.map_contrast_limits['tomo'] = None

        dirpath = Path(dirpath)
        tomo_fps = list(Path(dirpath).glob(pattern))

        self.map_ids['tomo'] = [str(x) for x in range(len(tomo_fps))]
        self.map_items_dict['tomo'] = {str(idx): dict(MapFile=str(tomo_fp)) for idx, tomo_fp in enumerate(tomo_fps)}

        self.match_to_parents['tomo'] = {
            'tomo': ['MapFile', match_self],
            self.get_parent_map_type('tomo'): match_parent
        }
        self.match_ids['tomo'] = self._get_map_match_ids('tomo')
        self._build_match_dicts_for_map('tomo')
        self.key_hierarchy['tomo'] = self._get_key_dependencies_for_map_type('tomo')
        self.map_filepaths['tomo'] = dict(zip(self.map_ids['tomo'], tomo_fps))
        # self.map_resolutions['tomo'] = self._get_map_resolutions('tomo')

        # With the tomograms matched to their respective parents, populate the view items with additional information
        tomo_parent = self.get_parent_map_type('tomo')
        self.map_shapes['tomo'] = self._get_map_shapes('tomo')
        self.map_resolutions['tomo'] = dict()
        for k, v in self.match_dict_bkw['tomo'].items():
            self.map_items_dict['tomo'][k]['MapScaleMat'] = self.map_items_dict[tomo_parent][v]['MapScaleMat']
            self.map_items_dict['tomo'][k]['StageXYZ'] = self.map_items_dict[tomo_parent][v]['StageXYZ']
            # Using the parent map resolution to ensure that the affine transformation is computed correctly
            self.map_resolutions['tomo'][k] = self.map_resolutions[tomo_parent][v]
