
import os


MOBIE_TABLE_ENTRY_NAMES = [
    'uri',
    'name',
    'type',
    'view',
    'group',
    'affine',
    'blend',
    'exclusive',
    'contrast_limits'
]


def get_mobie_table_path(project_path):
    return os.path.join(project_path, 'mobie.csv')


def normalize_mobie_table_entry_dict(entries, entry_count=None, use_abs_path=False, table_filepath=None):

    def _normalize_uri(uri):
        if use_abs_path:
            uri = os.path.abspath(uri)
        else:
            assert table_filepath is not None
            uri = os.path.relpath(uri, os.path.split(table_filepath)[0])
        return uri

    def _normalize_affine(affine):
        return ','.join([str(x) for x in affine])

    def _normalize_exclusive(exclusive):
        if type(exclusive) is not str:
            return 'true' if exclusive else 'false'
        return exclusive

    def _normalize_contrast_limits(contrast_limits):
        if contrast_limits == 'auto':
            return contrast_limits
        return ','.join(f'{x:.1f}' if isinstance(x, int) or x == int(x) else str(x) for x in contrast_limits)

    # Let's make sure everything is a list!
    for k, v in entries.items():
        assert type(v) == list, 'Entries have to be given as as list even if only one element is added!'

    entry_count_from_entries = max([len(v) for _, v in entries.items()])
    entry_count = entry_count_from_entries if entry_count is None else entry_count

    # First pass to determine the number of entries or check that the length of the arrays matches the given entry count
    for k, v in entries.items():

        assert len(v) == entry_count or len(v) == 1, 'The number of elements in an entry must match the entry count or be of len = 1'
        if len(v) == 1:
            entries[k] = v * entry_count

    # Now we can make sure that everything is properly readable by MoBIE
    for k, v in entries.items():

        for idx, item in enumerate(v):
            if k == 'uri':
                v[idx] = _normalize_uri(item)
            elif k == 'name':
                pass
            elif k == 'type':
                pass
            elif k == 'view':
                pass
            elif k == 'group':
                pass
            elif k == 'affine':
                v[idx] = _normalize_affine(item)
            elif k == 'blend':
                pass
            elif k == 'exclusive':
                v[idx] = _normalize_exclusive(item)
            elif k == 'contrast_limits':
                v[idx] = _normalize_contrast_limits(item)
            else:
                pass

        entries[k] = v

    return entries


def append_mobie_table(table_filepath, entries, use_abs_path=False):

    entries = normalize_mobie_table_entry_dict(entries, use_abs_path=use_abs_path, table_filepath=table_filepath)

    import pandas as pd

    table_data = pd.DataFrame()
    if os.path.exists(table_filepath):
        table_data = pd.read_csv(table_filepath, sep='\t')

    new_table_data = pd.concat([table_data, pd.DataFrame(entries)], ignore_index=True, sort=False)
    new_table_data = new_table_data.fillna('')

    new_table_data.to_csv(table_filepath, index=False, sep='\t')


def remove_mobie_table_entry(table_filepath, value, row='uri', verbose=False, debug=False):

    import pandas as pd

    table_data = pd.read_csv(table_filepath, sep='\t')
    if verbose:
        print(table_data)

    table_data = table_data[table_data[row] != value]
    table_data = table_data.fillna('')

    if verbose:
        print(table_data)

    if not debug:
        table_data.to_csv(table_filepath, index=False, sep='\t')


def replace_mobie_table(table_filepath, entries):

    import pandas as pd

    table_data = pd.DataFrame(entries)
    table_data.to_csv(table_filepath, index=False, sep='\t')


def update_mobie_table_entry(table_filepath, entry, item, allow_multiple=False):
    assert len(entry) == 2
    assert len(item) == 2

    import pandas as pd
    mobie_table = pd.read_csv(table_filepath, sep='\t')

    matches = mobie_table[mobie_table[item[0]] == item[1]].index

    if not allow_multiple:
        assert len(matches) == 1, "Multiple matches found, set allow_multiple=True"
        mobie_table.at[matches[0], entry[0]] = entry[1]
    else:
        mobie_table.loc[matches, entry[0]] = entry[1]

    mobie_table.to_csv(table_filepath, index=False, sep='\t')


def init_mobie_table(
        table_filepath,
        data_map_filepaths,
        use_abs_path=False,
        allow_any=False,
        verbose=False,
        **entry_kwargs
):
    if verbose:
        print(f'table_filepath = {table_filepath}')
        print(f'data_map_filepaths = {data_map_filepaths}')

    if os.path.exists(table_filepath):
        os.remove(table_filepath)

    if not allow_any:
        for k, v in entry_kwargs.items():
            assert k in MOBIE_TABLE_ENTRY_NAMES, f'{k} not in {MOBIE_TABLE_ENTRY_NAMES}'

    entries = dict(
            uri=data_map_filepaths,
            **entry_kwargs
    )

    append_mobie_table(
        table_filepath,
        entries,
        use_abs_path
    )


class MobieTableProject:

    def __init__(self, project_dirpath, verbose=False):

        self._project_dirpath = project_dirpath
        self._verbose = verbose
        self._mobie_table_path = get_mobie_table_path(self._project_dirpath)

        if not os.path.exists(project_dirpath):
            os.mkdir(project_dirpath)
            return

    def init_with_data_maps(
            self,
            data_map_filepaths,
            types='intensities',
            views='raw',
            groups='group0'
    ):
        assert len(os.listdir(self._project_dirpath)) == 0, 'Can only initialize a project in an empty directory'
        init_mobie_table(
            self._mobie_table_path,
            data_map_filepaths,
            types=types,
            views=views,
            groups=groups,
            verbose=self._verbose
        )

    def get_project_dirpath(self):
        return self._project_dirpath



