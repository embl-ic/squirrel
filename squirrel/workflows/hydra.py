
import numpy as np


def get_acquisition_transforms_workflow(dirpath: str, out_filepath: str = None, verbose: bool = False):

    if verbose:
        print(f'dirpath = {dirpath}')
        print(f'out_filepath = {out_filepath}')
        
    # Extract the slices geometries
    from squirrel.library.hydra import extract_dataset_geometry
    origins, shapes = extract_dataset_geometry(dirpath)

    # Convert to an affine sequence
    from squirrel.library.affine_matrices import AffineMatrix, AffineStack
    # transforms = AffineStack(stack=[AffineMatrix(translation=-x) for x in origins], is_sequenced=True)
    transforms = AffineStack(matrices=[AffineMatrix.from_translation(-x) for x in origins], sequenced=True)

    # Apply autopadding 
    stack_bounds = np.concatenate((np.zeros(shapes.shape), shapes), axis=1)
    transforms.set_metadata('bounds', stack_bounds)
    transforms, _ = transforms.auto_pad(extra_padding=16)

    if out_filepath is not None:
        transforms.write(out_filepath)

    return transforms


if __name__ == '__main__':

    dp = '/media/julian/Data/projects/hennies/squirrel-devel/hydra/data/'
    out_fp = '/media/julian/Data/projects/hennies/squirrel-devel/hydra/transforms.json'
    get_hydra_slice_geometry_workflow(dp, out_filepath=out_fp)

    from squirrel.workflows.transformation import apply_stack_alignment_on_volume_workflow

