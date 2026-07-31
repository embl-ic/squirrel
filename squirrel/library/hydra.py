
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
    
