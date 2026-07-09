
import os

from tifffile import imwrite, imread
import numpy as np
from glob import glob
from squirrel.library.data import invert_data


def make_directory(directory, exist_ok=False, not_found_ok=False):
    try:
        os.mkdir(directory)
    except FileNotFoundError:
        if not_found_ok:
            return 'not_found'
        raise
    except FileExistsError:
        if exist_ok:
            return 'exists'
        raise
    return None


def load_h5_container(filepath, key, axes_order='zyx', invert=False):

    from h5py import File

    with File(filepath, mode='r') as f:
        data = f[key][:]

    if invert:
        data = invert_data(data)

    if axes_order == 'zyx':
        return data

    axes_order_list = list(axes_order)
    return np.transpose(data, axes=[
        axes_order_list.index('z'),
        axes_order_list.index('y'),
        axes_order_list.index('x')
    ])


def write_h5_container(filepath, data, key='data', append=False):
    # TODO add test

    from h5py import File

    mode = 'a' if append else 'w'

    with File(filepath, mode=mode) as f:
        f.create_dataset(key, data=data, compression='gzip')


def write_n5_container(filepath, data, key='data'):

    from z5py import File

    with File(filepath, mode='w') as f:
        f.create_dataset(key, data=data, compression='gzip', chunks=(1, 1024, 1024))


def write_tif_stack(data, out_folder, id_offset=0, slice_name='slice_{:04d}.tif'):

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    for idx, section in enumerate(data):
        write_tif_slice(section, out_folder, slice_name.format(idx + id_offset))


def read_tif_slice(filepath, return_filepath=True):

    image = imread(filepath)

    # Return the image and the filename
    if return_filepath:
        return image, os.path.split(filepath)[1]
    return image


def read_png_slice(filepath, return_filepath=True):

    from skimage import io as sk_io

    image = sk_io.imread(filepath)

    # Return the image and the filename
    if return_filepath:
        return image, os.path.split(filepath)[1]
    return image


def write_tif_slice(image, out_folder=None, filename=None, out_filepath=None):
    if not bool(out_folder is None) ^ bool(out_filepath is None):
        raise ValueError('Supply either out_folder and filename or out_filepath, and not both!')
    if out_folder is not None:
        im_filepath = os.path.join(out_folder, filename)
    imwrite(im_filepath, image, compression='zlib')


def get_file_list(path, pattern='*'):
    return sorted(glob(os.path.join(path, pattern)))


# NOTE: Deprecated
# def load_tif_stack(path, pattern='*'):
#     return [read_tif_slice(filepath, return_filepath=False) for filepath in get_file_list(path, pattern)]


def write_nii_file(data, filepath, scale=None):
    # FIXME add tests
    import nibabel as nib
    affine = np.eye(4)
    if scale is not None:
        affine = np.eye(4)
        affine[0, 2] = scale[0]
        affine[1, 2] = scale[1]
        affine[2, 2] = scale[2]
    data = nib.Nifti1Image(data, affine=affine)
    nib.save(data, filepath)


def load_nii_file(filepath, invert=False):
    # FIXME add tests
    import nibabel as nib
    data = np.array(nib.load(filepath).dataobj)
    if invert:
        return invert_data(data)
    return data


def get_filetype(filepath):
    # FIXME add tests

    h5_extensions = ['.h5', '.H5', '.hdf5', '.HDF5']
    nii_extensions = ['.nii']
    tif_extensions = ['.tif', '.TIF', '.tiff', '.TIFF']
    json_extensions = ['.json', '.JSON']
    csv_extensions = ['.csv', '.CSV']
    zarr_extensions = ['.zarr', '.ZARR', '.zarr/', '.ZARR/']
    n5_extensions = ['.n5', '.N5', '.n5/', '.N5/']
    ome_zarr_sub_extensions = ['.ome', '.OME']

    basename, ext = os.path.splitext(filepath.strip("/"))
    sub_ext = ''
    if ext in h5_extensions:
        return 'h5'
    if ext in nii_extensions:
        return 'nii'
    if ext in tif_extensions:
        return 'tif'
    if ext in json_extensions:
        return 'json'
    if ext in csv_extensions:
        return 'csv'
    if ext in n5_extensions:
        return 'n5'
    if ext in zarr_extensions:
        sub_ext = os.path.splitext(basename)[1]
        if sub_ext in ome_zarr_sub_extensions:
            return 'ome_zarr'
    if ext == '':
        return 'dir'
    if os.path.isdir(filepath):
        return 'dir'
    raise ValueError(f'Unknown file extension: {sub_ext}.{ext}')


# NOTE: potentially deprecated (bring back if necessary)
# def load_data(filepath, key='data', axes_order='zyx', invert=False):
#
#     filetype = get_filetype(filepath)
#
#     if filetype == 'h5':
#         return load_h5_container(filepath, key, axes_order=axes_order, invert=invert)
#     if filetype == 'nii':
#         return load_nii_file(filepath, invert=invert)
#     if filetype == 'tif':
#         raise NotImplementedError('Not implemented for 3D tif files')
#     if filetype == 'dir':
#         return np.array(load_tif_stack(filepath))
#     raise RuntimeError(f'Invalid or unknown file type: {filetype}')


# NOTE: Deprecated
# def _load_data(h, idx):
#
#     if type(h[idx]) == str:
#         assert get_filetype(h[idx]) == 'tif'
#         return read_tif_slice(h[idx])
#
#     # Assuming h5, n5 or ome-zarr file handle
#     return h[idx], None


def get_reshaped_data(h, idx, shape):
    """
    :param h:
    :param idx:
    :param shape: Returns that data in the specified shape
    :return: np.array(data_slice), slice_filepath
    Note that slice_filepath is None if it's a hdf5 file handle
    """
    assert isinstance(idx, int)
    from squirrel.library.image import image_to_shape
    return image_to_shape(h[idx], shape)


def load_data_handle(path, key=None, pattern=None):

    filetype = get_filetype(path)

    if filetype == 'h5':
        from h5py import File
        h = File(path, mode='r')[key if key is not None else 'data']
        return h, h.shape

    if filetype == 'n5':
        from z5py import File
        h = File(path, mode='r')[key if key is not None else 'setup0/timepoint0/s0']
        return h, h.shape

    if filetype == 'dir':
        # default pattern: '*.tif' or '*.png'
        if pattern is None:
            # check if directory has any PNGs
            from pathlib import Path
            p = Path(path)
            if any(f.suffix.lower() == ".png" for f in p.iterdir()):
                h = PngStack(path)
                return h, h.get_shape()
            else:
                h = TiffStack(path, pattern="*.tif")
                return h, h.get_shape()
        else:
            # use user-provided pattern
            if pattern.lower().endswith(".png"):
                h = PngStack(path, pattern=pattern)
                return h, h.get_shape()
            else:
                h = TiffStack(path, pattern=pattern)
                return h, h.get_shape()

    if filetype == 'nii':
        h = load_nii_file(path)
        return h, h.shape

    if filetype == 'ome_zarr':
        from squirrel.library.ome_zarr import get_ome_zarr_handle
        h = get_ome_zarr_handle(path, key if key is not None else 's0', 'r')
        return h, h.shape

    raise RuntimeError(f'No valid filetype: {filetype}')


def crop_roi(h, roi):
    """

    :param h:
    :param roi: [min_x, max_x, min_y, max_y, z]
    :return:
    """

    assert len(roi) == 5
    roi = np.array(roi).astype(int)
    min_x, min_y, max_x, max_y, z = roi

    return h[z][min_y: max_y, min_x: max_x]


class _GenericStack(list):

    def __init__(self, dirpath, pattern=None, image_type='tif'):
        stack = get_file_list(dirpath, pattern)
        list.__init__(self, stack)
        self.image_type = image_type
        self.dtype = self[0].dtype
        self.shape = self.get_shape()
        self.chunks = [1] + self.shape[1:]

    def read_slice(self, filepath, return_filepath=True):
        if self.image_type == 'tif':
            return read_tif_slice(filepath, return_filepath)
        if self.image_type == 'png':
            return read_png_slice(filepath, return_filepath)

    def __getitem__(self, item):

        if isinstance(item, tuple):
            filepaths = [list.__getitem__(self, x) for x in item]
        else:
            filepaths = list.__getitem__(self, item)
        if isinstance(filepaths, str):
            return self.read_slice(filepaths, return_filepath=False)
        if isinstance(filepaths, list):
            stack = [self.read_slice(x, return_filepath=False) for x in filepaths]
            try:
                return np.array(stack)
            except ValueError:
                print(f"Warning: Inconsistent slice shapes! Can't convert to np.array, so returning list instead")
                return stack

    def __iter__(self):
        for x in super().__iter__():
            yield self.read_slice(x, return_filepath=False)

    def get_slice_and_filepath(self, idx):

        assert isinstance(idx, int)
        return self.read_slice(list.__getitem__(self, idx), return_filepath=True)

    def get_filepaths(self):
        return list.__getitem__(self, np.s_[:])

    def get_shape(self, check_all=False):
        """
        Returns the shape of the stack (z, y, x)
        :param check_all: If False, only the shape of the first slice is checked and inferred for x and y
            In case the stack has inconsisten slice shapes, use check_all=True which will return the max x and y values
        :return:
        """
        if check_all:
            st = self[:]
            if type(st) is list:
                shapes = []
                for idx, sl in enumerate(st):
                    shapes.append(sl.shape)
                shape = np.max(shapes, axis=0)
            else:
                shape = st[0].shape
            del st
        else:
            shape = self[0].shape
        return [len(self)] + list(shape)


class TiffStack(_GenericStack):
    def __init__(self, dirpath, pattern='*.tif'):
        _GenericStack.__init__(self, dirpath, pattern=pattern, image_type='tif')


class PngStack(_GenericStack):
    def __init__(self, dirpath, pattern='*.png'):
        _GenericStack.__init__(self, dirpath, pattern=pattern, image_type='png')


def write_stack(path, data, key='data', id_offset=0):

    # if os.path.splitext(path)[1] == '.h5':
    filetype = get_filetype(path)
    if filetype == 'h5':
        write_h5_container(path, data, key=key)
        return
    if filetype == 'dir':
        write_tif_stack(data, path, id_offset=id_offset)
        return
    if filetype == 'n5':
        write_n5_container(path, data, key=key)
        return
    if filetype == 'ome_zarr':
        from squirrel.library.ome_zarr import create_ome_zarr
        ozh = create_ome_zarr(path, data.shape, chunk_size=(128, 128, 128))
        ozh['s0'][:] = data
        return
    raise ValueError(f'Invalid filetype={filetype} of target path={path}')


if __name__ == '__main__':
    fp = '/media/julian/Data/projects/kors/align/4T/subset_6318/'
    ts = TiffStack(fp)
    print(ts[0: 10].shape)
    print(ts.get_shape())

