
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

TIFF_EXTENSIONS = {".tif", ".tiff"}


def _xml_float(
    root: ET.Element,
    path: str,
    *,
    required: bool = True,
) -> float:
    """Read a numeric XML field."""
    element = root.find(path)

    if element is None or element.text is None:
        if required:
            raise ValueError(f"Missing metadata field: {path}")
        return float("nan")

    try:
        return float(element.text)
    except ValueError as exc:
        raise ValueError(
            f"Metadata field {path!r} is not numeric: {element.text!r}"
        ) from exc


def _read_fei_metadata_root(tiff_path: Path) -> ET.Element:
    """Read the FEI/Thermo Fisher XML stored in TIFF tag 0x877B."""
    import tifffile
    with tifffile.TiffFile(tiff_path) as tif:
        if not tif.pages:
            raise ValueError("TIFF contains no image pages")

        tag = tif.pages[0].tags.get(0x877B)

        if tag is None:
            raise ValueError("FEI metadata tag 0x877B is missing")

        xml_data = tag.value

    if isinstance(xml_data, bytes):
        xml_data = xml_data.decode("utf-8", errors="strict")

    # TIFF string tags sometimes contain trailing null characters.
    xml_data = xml_data.rstrip("\x00")

    return ET.fromstring(xml_data)



def _xml_element_to_dict(element: ET.Element):
    """Convert an XML element to JSON-compatible Python values.

    Attributes use ``@`` prefixes and mixed text uses ``#text``.
    Repeated child tags become lists. Leaf text is preserved as a string.
    """
    result = {f"@{key}": value for key, value in element.attrib.items()}

    for child in element:
        value = _xml_element_to_dict(child)

        if child.tag not in result:
            result[child.tag] = value
        elif isinstance(result[child.tag], list):
            result[child.tag].append(value)
        else:
            result[child.tag] = [result[child.tag], value]

    text = (element.text or "").strip()

    if result:
        if text:
            result["#text"] = text
        return result

    return text if text else None


def read_hydra_metadata(tiff_path: str | Path) -> dict:
    """Read the complete FEI/Hydra XML metadata from a TIFF file.

    The XML root tag is retained as the outer dictionary key. Values are
    kept as strings to avoid guessing the types or units of raw metadata.
    """
    root = _read_fei_metadata_root(Path(tiff_path).expanduser())
    return {root.tag: _xml_element_to_dict(root)}


# Named fields are extracted from the original XML to retain explicit units.
# Values are returned in the units declared here.
HYDRA_METADATA_FIELDS = {
    'pixel_size_x': {'path': './BinaryResult/PixelSize/X', 'unit': 'nm', 'scale': 1e9},
    'pixel_size_y': {'path': './BinaryResult/PixelSize/Y', 'unit': 'nm', 'scale': 1e9},
    'slice_thickness': {'path': 'VoxelSizeInfo/VoxelZ', 'unit': 'nm'},
    'stage_x': {'path': './StageSettings/StagePosition/X', 'unit': 'm'},
    'stage_y': {'path': './StageSettings/StagePosition/Y', 'unit': 'm'},
    'pretilt_angle': {'path': './Optics/SamplePreTiltAngle', 'unit': 'rad'},
}

HYDRA_METADATA_PRESETS = {
    'resolution': ('pixel_size_x', 'pixel_size_y', 'slice_thickness'),
    'geometry': ('pixel_size_x', 'pixel_size_y', 'stage_x', 'stage_y', 'pretilt_angle'),
}


def _read_named_hydra_field(root: ET.Element, field: str) -> float | None:
    """Read one named field, returning None when it is absent."""
    definition = HYDRA_METADATA_FIELDS[field]
    path = definition['path']

    if path.startswith('VoxelSizeInfo/'):
        # Voxel sizes are stored in a scoped custom XML section.
        element = root.find(
            "./CustomSectionGroup/CustomSection[@scope='VoxelSizeInfo']/" + path
        )
    else:
        element = root.find(path)

    if element is None or element.text is None or not element.text.strip():
        return None

    value = element.text.strip()

    if path.startswith('VoxelSizeInfo/'):
        # These fields contain an explicit unit, e.g. '15 nm'.
        parts = value.split()
        if len(parts) != 2 or parts[1] != definition['unit']:
            raise ValueError(f"Unexpected unit for {field}: {value!r}")
        value = parts[0]

    try:
        return float(value) * definition.get('scale', 1.0)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric metadata for {field}: {value!r}") from exc


def extract_hydra_metadata(
    tiff_path: str | Path,
    preset: str | None = None,
    fields: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Extract named Hydra metadata fields with values and units.

    Preset fields and explicit fields are combined in insertion order.
    Missing fields have a null value; unknown field names raise ValueError.
    """
    if preset is not None and preset not in HYDRA_METADATA_PRESETS:
        raise ValueError(f"Unknown Hydra metadata preset: {preset!r}")

    selected = list(HYDRA_METADATA_PRESETS[preset]) if preset else []
    selected.extend(fields or [])
    selected = list(dict.fromkeys(selected))

    for field in selected:
        if field not in HYDRA_METADATA_FIELDS:
            raise ValueError(f"Unknown Hydra metadata field: {field!r}")

    root = _read_fei_metadata_root(Path(tiff_path).expanduser())

    return {
        field: {
            'value': _read_named_hydra_field(root, field),
            'unit': HYDRA_METADATA_FIELDS[field]['unit'],
        }
        for field in selected
    }


def _extract_origin_and_shape(
    tiff_path: Path,
) -> tuple[list[float], list[int]]:
    """
    Extract the global image origin and stored image shape.

    Returns
    -------
    origin
        [global_origin_y_px, global_origin_x_px]

    shape
        [img_shape_y, img_shape_x]
    """
    root = _read_fei_metadata_root(tiff_path)

    scan_x = _xml_float(root, "./ScanSettings/ScanArea/X")
    scan_y = _xml_float(root, "./ScanSettings/ScanArea/Y")

    scan_width = _xml_float(root, "./ScanSettings/ScanArea/Width")
    scan_height = _xml_float(root, "./ScanSettings/ScanArea/Height")

    scan_size_width = _xml_float(
        root,
        "./ScanSettings/ScanSize/Width",
    )
    scan_size_height = _xml_float(
        root,
        "./ScanSettings/ScanSize/Height",
    )

    pixel_size_x = _xml_float(
        root,
        "./BinaryResult/PixelSize/X",
    )
    pixel_size_y = _xml_float(
        root,
        "./BinaryResult/PixelSize/Y",
    )

    stage_x_m = _xml_float(
        root,
        "./StageSettings/StagePosition/X",
    )
    stage_y_m = _xml_float(
        root,
        "./StageSettings/StagePosition/Y",
    )

    pretilt_rad = _xml_float(
        root,
        "./Optics/SamplePreTiltAngle",
    )

    if pixel_size_x == 0 or pixel_size_y == 0:
        raise ValueError("Pixel size must be nonzero")

    cos_pretilt = math.cos(pretilt_rad)

    if math.isclose(cos_pretilt, 0.0, abs_tol=1e-12):
        raise ValueError(
            "Cannot apply Y tilt correction because cos(pretilt) is zero"
        )

    # Position of the stored image's upper-left corner relative to
    # the center of the full scan raster.
    raster_origin_x_px = scan_x - scan_size_width / 2.0
    raster_origin_y_px = scan_y - scan_size_height / 2.0

    # Convert absolute stage coordinates from metres to image pixels.
    stage_x_px = stage_x_m / pixel_size_x
    stage_y_tilt_corrected_px = (
        stage_y_m / pixel_size_y / cos_pretilt
    )

    global_origin_x_px = raster_origin_x_px + stage_x_px
    global_origin_y_px = (
        raster_origin_y_px - stage_y_tilt_corrected_px
    )

    origin = [
        global_origin_y_px,
        global_origin_x_px,
    ]

    shape = [
        int(round(scan_height)),
        int(round(scan_width)),
    ]

    return origin, shape


def extract_dataset_geometry(
    dirpath: str | Path
) -> tuple[list[list[float]], list[list[int]]]:
    """
    Parse all TIFF slices in a directory.

    Parameters
    ----------
    dirpath
        Directory containing the TIFF slices.

    skip_invalid
        Skip TIFFs with missing or invalid metadata when True.
        Otherwise, raise an exception.

    Returns
    -------
    origins
        List of [global_origin_y_px, global_origin_x_px].

    shapes
        List of [img_shape_y, img_shape_x].

    Notes
    -----
    The returned lists are ordered alphabetically by filename.
    """
    dirpath = Path(dirpath).expanduser().resolve()

    if not dirpath.is_dir():
        raise NotADirectoryError(f"Not a directory: {dirpath}")

    tiff_paths = sorted(
        (path for path in dirpath.iterdir() if path.is_file() and path.suffix.lower() in TIFF_EXTENSIONS),
        key=lambda p: p.name,
    )

    if not tiff_paths:
        raise FileNotFoundError(f"No TIFF files found in {dirpath}")

    origins: list[list[float]] = []
    shapes: list[list[int]] = []

    for tiff_path in tiff_paths:
        try:
            origin, shape = _extract_origin_and_shape(tiff_path)
        except Exception as exc:
            raise RuntimeError(f"Could not parse metadata from {tiff_path}") from exc

        origins.append(origin)
        shapes.append(shape)

    origins = np.asarray(origins, dtype=float) 
    shapes = np.asarray(shapes, dtype=int) 

    return origins, shapes


if __name__ == '__main__':

    dp = '/media/julian/Data/projects/hennies/squirrel-devel/hydra/data/'

    oris, shps = extract_dataset_geometry(dp)

    print(f'oris = {oris}')
    print(f'shps = {shps}')
    
