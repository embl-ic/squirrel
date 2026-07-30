# squirrel

Python-based library for conversion and processing of 3D EM data.

## Installation via conda

hopefully coming soon:

```
conda create -n squirrel-env squirrel
```

## Installation via pip

```
mamba create -n squirrel-env -c conda-forge python=3.11
mkdir path/to/src
cd path/to/src
git clone https://github.com/jhennies/squirrel
pip install -e squirrel
```

## Installation of optional packages

Install the following as necessary to enable additional functionality

### nibabel for handling nii files
This is required for 3D SIFT
```
mamba install -c conda-forge nibabel
```

### Napari for 3D visualization
```
mamba install -c conda-forge napari pyqt
```

### For registration
```
pip install SimpleITK-SimpleElastix
pip install transforms3d
```
also install OpenCV (see below)

### OpenCV for 2D SIFT 
```
mamba install -c conda-forge opencv
```

### For ome.zarr support
```
mamba install -c conda-forge zarr=2
```

### For faster filters or morphological operations

Note: this changes the numpy version!
```
mamba install -c conda-forge vigra
```

### For MoBIE project support
```
mamba install -c conda-forge pandas
```

### For VSNR

```
pip install pyvsnr
```

## Specific installations

### PyVista

```bash 
mamba create -n pyvista-squirrel-env -c conda-forge --override-channels python=3.11 "numpy<2.3" vigra pyvista tol-colors
conda activate pyvista-squirrel-env
pip install squirrel
```

## Usage

The main functions can be used directly after activation of the 
conda environment:

```
conda activate squirrel-env

sq-conversion-cast_dtype -h
sq-mobie-export_rois_with_mobie_table -h
sq-serialem-parse_navigator_file -h
sq-stack-clahe_on_stack -h
sq-stack-estimate_crop_xy -h
sq-stack-invert_slices -h
sq-conversion-cast_segmentation -h
sq-mobie-init_project -h
sq-stack-axis_median_filter -h
sq-stack-compress_tif_stack -h
sq-stack-filter_2d_workflow -h
sq-stack-normalize_slices -h
sq-conversion-n5_to_stack -h
sq-serialem-create_link_maps -h
sq-stack-calculator -h
sq-stack-crop_from_stack -h
sq-stack-get_label_list -h
sq-stack-tif_nearest_scaling -h
```

The -h flag yields the help output describing functions and parameters
