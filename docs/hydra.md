# Hydra TIFF metadata

Squirrel reads FEI/Thermo Fisher metadata embedded in Hydra SEM TIFF files. The
metadata is stored as XML in TIFF tag `0x877B`. The reader inspects TIFF tags;
it does not need to load the image pixel array.

## Command-line interface

Install squirrel in the active Python environment (for example, `pip install -e .`)
to register `sq-hydra-extract_tif_metadata`.

Print the complete metadata to the terminal as YAML:

```bash
sq-hydra-extract_tif_metadata input.tif
```

Print a preset of interpreted fields:

```bash
sq-hydra-extract_tif_metadata input.tif --preset resolution
```

Print selected fields, or combine them with a preset:

```bash
sq-hydra-extract_tif_metadata input.tif --fields pixel_size_x slice_thickness
sq-hydra-extract_tif_metadata input.tif --preset resolution --fields stage_x stage_y
```

Save the YAML instead of printing it by specifying `-o` or `--out_filepath`:

```bash
sq-hydra-extract_tif_metadata input.tif --preset resolution -o resolution.yaml
sq-hydra-extract_tif_metadata input.tif --out_filepath complete.yaml
```

Without `--preset` or `--fields`, the command exports the full XML-derived
metadata tree. When both selection options are provided, the selected fields
are combined in order without duplicates.

## Named fields and presets

Named fields have a numeric `value` (or `null` if absent) and a `unit`.

| Field | Unit | Source in Hydra XML |
| --- | --- | --- |
| `pixel_size_x` | nm | `BinaryResult/PixelSize/X` (stored in m) |
| `pixel_size_y` | nm | `BinaryResult/PixelSize/Y` (stored in m) |
| `slice_thickness` | nm | `CustomSectionGroup/CustomSection[@scope='VoxelSizeInfo']/VoxelSizeInfo/VoxelZ` |
| `stage_x` | m | `StageSettings/StagePosition/X` |
| `stage_y` | m | `StageSettings/StagePosition/Y` |
| `pretilt_angle` | rad | `Optics/SamplePreTiltAngle` |

The `resolution` preset selects `pixel_size_x`, `pixel_size_y`, and
`slice_thickness`. The `geometry` preset selects `pixel_size_x`,
`pixel_size_y`, `stage_x`, `stage_y`, and `pretilt_angle`. Preset definitions
and field definitions are available as `HYDRA_METADATA_PRESETS` and
`HYDRA_METADATA_FIELDS` in `squirrel.library.hydra`.

For example, a resolution export might look like:

```yaml
pixel_size_x:
  value: 7.0
  unit: nm
pixel_size_y:
  value: 7.0
  unit: nm
slice_thickness:
  value: 15.0
  unit: nm
```

## Python API

```python
from squirrel.library.hydra import (
    extract_hydra_metadata,
    read_hydra_metadata,
    save_hydra_metadata,
)

raw_metadata = read_hydra_metadata('input.tif')
resolution = extract_hydra_metadata('input.tif', preset='resolution')
selected = extract_hydra_metadata(
    'input.tif', fields=['pixel_size_x', 'slice_thickness']
)
save_hydra_metadata(resolution, 'resolution.yaml')
```

`read_hydra_metadata()` returns a nested dictionary with the XML root tag
(typically `Metadata`) as the outer key. Leaf values remain strings, repeated
XML elements become lists, attributes are prefixed with `@`, and mixed text
uses `#text`. Empty leaf elements become `None`. This is a representation of
the XML, not an interpretation of its physical units.

`extract_hydra_metadata()` reads the original XML and returns only the
requested named fields with their interpreted units. Missing known fields
have `value: null`; unknown field or preset names raise `ValueError`.
Unexpected units or nonnumeric values in selected fields also raise
`ValueError`. The `slice_thickness` field reads `VoxelZ` from the scoped
`VoxelSizeInfo` section; it is not inferred from the filename or slice index.

`save_hydra_metadata(metadata, out_filepath)` writes the supplied dictionary
as YAML, preserving dictionary key order. The CLI prints equivalent YAML when
no output path is provided.

## Dataset geometry extraction

The existing geometry API remains available:

```python
from squirrel.library.hydra import extract_dataset_geometry

origins, shapes = extract_dataset_geometry('path/to/tiff_directory')
```

It processes `.tif` and `.tiff` files in alphabetical filename order and
returns NumPy arrays of global image origins (`y`, `x`, in pixels) and stored
image shapes (`y`, `x`, in pixels). The origin calculation uses scan position,
stage coordinates, pixel sizes, and sample pretilt. This geometry calculation
is distinct from the named metadata extraction API.

## Errors and limitations

- A TIFF without FEI metadata tag `0x877B` raises `ValueError`.
- Malformed embedded XML raises `xml.etree.ElementTree.ParseError`.
- Named-field extraction requires a preset and/or a nonempty field selection;
  calling it with neither returns an empty dictionary. To read everything,
  use `read_hydra_metadata()`.
- This reader targets the Hydra/FEI XML layout. Other TIFF files may not
  contain the expected fields or units.
