# =============================================================================
# Imports
# =============================================================================

import os
import shutil

import numpy as np
import zarr


# =============================================================================
# Helper functions
# =============================================================================

def open_ome_zarr(path, mode="r"):
    """Open an OME-Zarr group."""
    return zarr.open(path, mode=mode)


# -----------------------------------------------------------------------------
# Version handling
# -----------------------------------------------------------------------------


def detect_ome_version(root):
    """
    Detect the OME-NGFF version.

    Returns
    -------
    str
        "0.4" or "0.5"
    """

    attrs = root.attrs

    if "multiscales" not in attrs:
        raise ValueError("Not an OME-Zarr dataset.")

    ms = attrs["multiscales"][0]

    return ms.get("version", "0.4")


def detect_zarr_format(root):
    """
    Detect the Zarr format version.

    Returns
    -------
    int
        2 or 3
    """

    metadata = getattr(root, "metadata", None)

    if metadata is None:
        return 2

    return getattr(metadata, "zarr_format", 2)


def validate_ome_version(version):

    version = str(version)

    if version not in ("0.4", "0.5"):
        raise ValueError(f"Unsupported OME version: {version}")

    return version


def validate_zarr_format(version):

    version = int(version)

    if version not in (2, 3):
        raise ValueError(f"Unsupported Zarr format: {version}")

    return version


def validate_downsample_method(method):

    method = method.lower()

    if method not in ("average", "sample"):
        raise ValueError(f"Unsupported downsampling method: {method}")

    return method.title()


# -----------------------------------------------------------------------------
# ROI utilities
# -----------------------------------------------------------------------------


def make_slice(position, shape):

    position = np.asarray(position)
    shape = np.asarray(shape)

    return tuple(slice(p, p + s) for p, s in zip(position, shape))


def roi_to_next_level(position, shape, factor):

    position = np.asarray(position)
    shape = np.asarray(shape)
    factor = np.asarray(factor)

    return (position // factor, shape // factor)


def expand_roi(position, shape, factor):

    position = np.asarray(position)
    shape = np.asarray(shape)
    factor = np.asarray(factor)

    start = (position // factor) * factor

    end = ((position + shape + factor - 1) // factor) * factor

    return start, end - start


def crop_roi(position, shape, dataset_shape):

    position = np.asarray(position)
    shape = np.asarray(shape)
    dataset_shape = np.asarray(dataset_shape)

    start = np.maximum(position, 0)

    end = np.minimum(position + shape, dataset_shape)

    return start, end - start


# -----------------------------------------------------------------------------
# Alignment
# -----------------------------------------------------------------------------

def storage_grid(dataset):
    """
    Return the storage grid of a dataset.

    v2  -> chunks
    v3  -> shards (if present), otherwise chunks
    """

    if hasattr(dataset, "shards"):

        shards = dataset.shards

        if shards is not None:
            return tuple(shards)

    return tuple(dataset.chunks)


def check_grid_alignment(position, shape, grid_shape, dataset_shape):

    position = np.asarray(position)
    shape = np.asarray(shape)
    grid_shape = np.asarray(grid_shape)
    dataset_shape = np.asarray(dataset_shape)

    if np.any(position % grid_shape):
        return False

    for p, s, g, ds in zip(position, shape, grid_shape, dataset_shape):

        if p + s == ds:
            continue

        if s % g:
            return False

    return True


# -----------------------------------------------------------------------------
# Pyramid utilities
# -----------------------------------------------------------------------------


def cumulative_downsample_factors(factors):

    cumulative = []

    current = np.ones(len(factors[0]), dtype=int)

    for factor in factors:

        current *= np.asarray(factor)
        cumulative.append(tuple(current))

    return cumulative


def compute_scales(resolution, downsample_factors):

    scales = [np.asarray(resolution, dtype=float)]

    current = scales[0].copy()

    for factor in downsample_factors:

        current = current * np.asarray(factor)
        scales.append(current.copy())

    return [s.tolist() for s in scales]


# =============================================================================
# Metadata
# =============================================================================


class OMEZarrMetadata:

    def __init__(self, root):

        self.root = root

        self.ome_version = detect_ome_version(root)
        self.zarr_format = detect_zarr_format(root)

        self.datasets = []
        self.axes = None
        self.downsample_method = None

        self._parse()

    # -------------------------------------------------------------------------
    # Creation
    # -------------------------------------------------------------------------

    @staticmethod
    def create(
            root, 
            downsample_factors=((2, 2, 2),), 
            resolution=(1., 1., 1.), 
            unit="micrometer", 
            downsample_method="Average", 
            ome_version="0.5", 
            zarr_format=3
    ):

        md = OMEZarrMetadata.__new__(OMEZarrMetadata)
        md.root = root

        md.ome_version = validate_ome_version(ome_version)
        md.zarr_format = validate_zarr_format(zarr_format)

        md.downsample_method = validate_downsample_method(downsample_method)

        md.axes = [
            {"name": "z", "type": "space", "unit": unit},
            {"name": "y", "type": "space", "unit": unit},
            {"name": "x", "type": "space", "unit": unit}
        ]

        scales = compute_scales(resolution, downsample_factors)

        md.datasets = []

        for idx, scale in enumerate(scales):
            md.datasets.append({"path": f"s{idx}", "scale": scale})

        md.write()
        return md

    # -------------------------------------------------------------------------
    # Parsing
    # -------------------------------------------------------------------------

    def _parse(self):

        if self.ome_version == "0.4":

            self._parse_ngff04()
            return

        if self.ome_version == "0.5":

            self._parse_ngff05()
            return

        raise RuntimeError(f"Unsupported NGFF version: {self.ome_version}")

    def _parse_ngff04(self):

        ms = self.root.attrs["multiscales"][0]

        self.axes = ms["axes"]
        self.downsample_method = ms.get("type", "Average")
        self.datasets = []

        for ds in ms["datasets"]:
            self.datasets.append({"path": ds["path"], "scale": ds["coordinateTransformations"][0]["scale"]})

    def _parse_ngff05(self):

        # Current implementation is identical.
        # If NGFF evolves further this function can diverge.
        self._parse_ngff04()

    # -------------------------------------------------------------------------
    # Writing
    # -------------------------------------------------------------------------

    def write(self):

        if self.ome_version == "0.4":

            self._write_ngff04()
            return

        if self.ome_version == "0.5":

            self._write_ngff05()
            return

        raise RuntimeError

    def _write_ngff04(self):

        datasets = []

        for ds in self.datasets:

            datasets.append({
                "path": ds["path"],
                "coordinateTransformations": [{"type": "scale", "scale": ds["scale"]}]
            })

        self.root.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": "/",
                "axes": self.axes,
                "datasets": datasets,
                "type": self.downsample_method,
            }
        ]

    def _write_ngff05(self):

        datasets = []

        for ds in self.datasets:

            datasets.append({
                "path": ds["path"],
                "coordinateTransformations": [{"type": "scale", "scale": ds["scale"]}]
            })

        self.root.attrs["multiscales"] = [
            {
                "version": "0.5",
                "name": "/",
                "axes": self.axes,
                "datasets": datasets,
                "type": self.downsample_method,
            }
        ]

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def levels(self):
        return [ds["path"] for ds in self.datasets]

    @property
    def scales(self):
        return [ds["scale"] for ds in self.datasets]

    @property
    def downsample_factors(self):

        factors = []
        prev = np.asarray(self.scales[0], dtype=float)

        for scale in self.scales[1:]:

            scale = np.asarray(scale, dtype=float)
            factors.append(tuple((scale / prev).astype(int)))
            prev = scale

        return factors

    @property
    def cumulative_downsample_factors(self):
        return cumulative_downsample_factors(self.downsample_factors)

    @property
    def units(self):
        units = [axis.get("unit") for axis in self.axes]
        assert len(set(units)) == 1
        return units[0]

    def storage_grid(self, dataset):
        return storage_grid(dataset)

    def scale(self, level):
        return tuple(self.datasets[level]["scale"])

# =============================================================================
# Pyramid generation
# =============================================================================


class PyramidBuilder:

    def __init__(self, store):
        self.store = store

    def _process_block(self, level, target_position, target_shape):

        factor = np.asarray(self.store.metadata.downsample_factors[level - 1])

        source_position = target_position * factor
        source_shape = target_shape * factor

        source = self.store.read(level - 1, source_position, source_shape)

        target = self.downsample(source, factor)

        self.store.write(level, target_position, target, update_pyramid=False)

    # -------------------------------------------------------------------------
    # Downsampling
    # -------------------------------------------------------------------------

    def downsample(self, array, factor):

        factor = np.asarray(factor)

        method = self.store.metadata.downsample_method.lower()

        if method == "sample":
            return array[tuple(slice(None, None, f) for f in factor)]

        if method == "average":

            # Pad to next multiple of factor.
            # This handles image boundaries cleanly.

            shape = np.asarray(array.shape)
            padded_shape = (np.ceil(shape / factor).astype(int) * factor)

            if np.any(shape != padded_shape):

                pad_width = [(0, p - s) for s, p in zip(shape, padded_shape)]
                array = np.pad(array, pad_width, mode="edge")
                shape = padded_shape

            new_shape = shape // factor

            reshape = []

            for n, f in zip(new_shape, factor):
                reshape.extend([n, f])

            axes = tuple(range(1, len(reshape), 2))

            return (array.reshape(reshape).mean(axis=axes).astype(array.dtype))

        raise ValueError(f"Unknown downsampling method: {method}")

    # -------------------------------------------------------------------------
    # Pyramid update
    # -------------------------------------------------------------------------

    def update_chunk(self, position, shape, source_level=0):

        position = np.asarray(position)
        shape = np.asarray(shape)

        for level in range(source_level + 1, len(self.store.metadata.levels)):

            factor = np.asarray(self.store.metadata.downsample_factors[level - 1])

            source_position, source_shape = expand_roi(position, shape, factor)

            source_position, source_shape = crop_roi(source_position, source_shape, self.store.shape(level - 1))

            target_position = source_position // factor
            target_shape = np.ceil(source_shape / factor).astype(int)

            grid = np.asarray(self.store.storage_grid(level))

            first = (target_position // grid) * grid
            last = ((target_position + target_shape - 1) // grid) * grid

            for z in range(first[0], last[0] + 1, grid[0]):
                for y in range(first[1], last[1] + 1, grid[1]):
                    for x in range(first[2], last[2] + 1, grid[2]):

                        p = np.array((z, y, x))
                        s = np.minimum(grid, np.asarray(self.store.shape(level)) - p)
                        self._process_block(level, p, s)

            position = target_position
            shape = target_shape

    # -------------------------------------------------------------------------
    # Full rebuild
    # -------------------------------------------------------------------------

    def rebuild(self, n_threads=1):

        from concurrent.futures import ThreadPoolExecutor

        for level in range(1, len(self.store.metadata.levels)):

            blocks = list(self.store.iter_storage_blocks(level))

            if n_threads == 1:

                for position, shape in blocks:
                    self._process_block(level, position, shape)

            else:

                with ThreadPoolExecutor(max_workers=n_threads) as tp:
                    list(tp.map(lambda b: self._process_block(level, *b), blocks))


# =============================================================================
# Main public API
# =============================================================================


class OMEZarrStore:

    def __init__(self, path, mode="r"):

        self.root = open_ome_zarr(path, mode)
        self.metadata = OMEZarrMetadata(self.root)
        self.pyramid = PyramidBuilder(self,)

    # -------------------------------------------------------------------------
    # Creation
    # -------------------------------------------------------------------------

    @staticmethod
    def create(
            path,
            shape,
            dtype=np.uint16,
            chunks=(64, 64, 64),
            shards=None,
            downsample_factors=((2, 2, 2),),
            resolution=(1., 1., 1.),
            unit="micrometer",
            downsample_method="Average",
            ome_version="0.4",
            zarr_format=2,
            overwrite=False,
    ):

        ome_version = validate_ome_version(ome_version)
        zarr_format = validate_zarr_format(zarr_format)

        import numbers
        if isinstance(chunks[0], numbers.Number):
            chunks = [list(chunks)] * (len(downsample_factors) + 1)
        if len(chunks) != len(downsample_factors) + 1:
            raise ValueError(f'Number of chunk sizes={len(chunks)} must match downsample factor count + 1={len(downsample_factors) + 1}!')

        if os.path.exists(path):
            if overwrite:
                shutil.rmtree(path)
            else:
                raise FileExistsError(path)

        root = zarr.open(path, mode="w", zarr_format=zarr_format)

        current_shape = np.asarray(shape)

        for level in range(len(downsample_factors) + 1):

            kwargs = dict(
                shape=tuple(current_shape),
                dtype=dtype,
                chunks=chunks[level],
                overwrite=True,
            )

            if zarr_format == 3 and shards is not None:
                kwargs["shards"] = shards

            root.create_array(f"s{level}", **kwargs)

            if level < len(downsample_factors):
                current_shape = np.maximum(1, current_shape // np.asarray(downsample_factors[level]))

        OMEZarrMetadata.create(
            root,
            downsample_factors=downsample_factors,
            resolution=resolution,
            unit=unit,
            downsample_method=downsample_method,
            ome_version=ome_version,
            zarr_format=zarr_format,
        )

        return OMEZarrStore(path, mode="r+")

    
    # -------------------------------------------------------------------------
    # Dataset access
    # -------------------------------------------------------------------------

    def dataset(self, level):
        return self.root[self.metadata.levels[level]]

    def shape(self, level):
        return self.dataset(level).shape

    def chunks(self, level):

        ds = self.dataset(level)

        if hasattr(ds, "chunks"):
            return tuple(ds.chunks)

        return None

    def shards(self, level):

        ds = self.dataset(level)

        if hasattr(ds, "shards"):
            return ds.shards

        return None

    def dtype(self, level):
        return self.dataset(level).dtype

    def storage_grid(self, level):
        return self.metadata.storage_grid(self.dataset(level))
    
    # -------------------------------------------------------------------------
    # Reading
    # -------------------------------------------------------------------------

    def read(self, level, position, shape):
        return self.dataset(level)[make_slice(position, shape)]

    # -------------------------------------------------------------------------
    # Alignment
    # -------------------------------------------------------------------------

    def check_alignment(self, level, position, shape):

        ds = self.dataset(level)

        if not check_grid_alignment(position, shape, self.metadata.storage_grid(ds), ds.shape):
            raise ValueError(f'ROI is not aligned to storage grid of level {level}.')

    def check_pyramid_alignment(self, level, position, shape):

        position = np.asarray(position)
        shape = np.asarray(shape)

        for lvl in range(level, len(self.metadata.levels)):

            self.check_alignment(lvl, position, shape)

            if lvl < len(self.metadata.downsample_factors):

                factor = np.asarray(self.metadata.downsample_factors[lvl])

                position //= factor
                shape //= factor

    # -------------------------------------------------------------------------
    # Writing
    # -------------------------------------------------------------------------

    def write(
            self, level, position, data, 
            update_pyramid=False, 
            check_alignment=False, 
            check_pyramid_alignment=False, 
            require_empty=False
    ):

        ds = self.dataset(level)

        position = np.asarray(position)
        shape = np.asarray(data.shape)

        if check_alignment:
            self.check_alignment(level, position.copy(), shape.copy())

        if check_pyramid_alignment:
            self.check_pyramid_alignment(level, position.copy(), shape.copy())

        sl = make_slice(position, shape)

        if require_empty:

            if np.any(ds[sl] != 0):
                raise ValueError("Attempting to overwrite existing data.")
        
        ds[sl] = data

        if update_pyramid:
            self.update_pyramid(position, shape, source_level=level)

    # -------------------------------------------------------------------------
    # Pyramid
    # -------------------------------------------------------------------------

    def update_pyramid(self, position, shape, source_level=0):
        self.pyramid.update_chunk(position, shape, source_level)

    def rebuild_pyramid(self, n_threads=1):
        self.pyramid.rebuild(n_threads=n_threads)

    def iter_storage_blocks(self, level):

        grid = np.asarray(self.storage_grid(level))

        shape = np.asarray(self.shape(level))

        starts = [range(0, s, g) for s, g in zip(shape, grid)]

        for z in starts[0]:
            for y in starts[1]:
                for x in starts[2]:

                    position = np.array((z, y, x))
                    block_shape = np.minimum(grid, shape - position)
                    yield (position, block_shape)


if __name__ == '__main__':

    # Example1: Create and write to an ome-zarr
    # Create an empty ome-zarr
    fp = '/media/julian/Data/tmp/create_ome_zarr_test.ome.zarr'
    oz = OMEZarrStore.create(
        path=fp,
        dtype='uint8',
        shape=(256, 256, 256),
        overwrite=True,
        ome_version='0.5',
        zarr_format=3,
        shards=(128, 128, 128)
    )
    # Fill it with data
    oz.write(
        level=0,
        data=np.ones((256, 256, 256)) * 255,
        position=(0, 0, 0),
        update_pyramid=True
    ) 

    # Example2: Write to an exisiting ome-zarr in multiple write events (e.g. sitting in different nextflow jobs)
    #    Note: the following write events can be done in parallel as they write one shard each

    # --- write event 1 ---
    oz = OMEZarrStore(fp, mode='a')
    oz.write(
        level=0,
        data=np.ones((128, 128, 128)) * 255,
        position=(0, 0, 0),
        check_alignment=True,  # Make sure that the written data fits the storage grid (chunks for version2, shards for version3)
        check_pyramid_alignment=False,  # This would ensure that the written data fits the storage grid through the entire pyramid (not nessesary if not computing the pyramid directly)
        update_pyramid=False
    )

    # --- write event 2 ---
    oz = OMEZarrStore(fp, mode='a')
    oz.write(
        level=0,
        data=np.ones((128, 128, 128)) * 255,
        position=(128, 0, 0),
        check_alignment=True, 
        check_pyramid_alignment=False,
        update_pyramid=False
    )

    # --- Finalizing the ome-zarr (after all writing is completed) ---
    oz.rebuild_pyramid(n_threads=os.cpu_count())

