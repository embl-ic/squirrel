
def sift_log_to_affine_stack_workflow(
        log_filepath,
        out_filepath=None,
        pivot=None,
        verbose=False
):

    if verbose:
        print(f'log_filepath = {log_filepath}')
        print(f'out_filepath = {out_filepath}')

    from squirrel.library.fiji import sift_log_to_affine_stack
    with open(log_filepath, mode='r') as f:
        transforms = sift_log_to_affine_stack(f.readlines(), pivot=pivot)

    from squirrel.library.affine_matrices import AffineStack
    out_transforms = AffineStack.identity(1, ndim=2, pivot=pivot, sequenced=False)
    out_transforms.extend(transforms)

    if out_filepath is not None:
        out_transforms.write(out_filepath)

    return out_transforms


if __name__ == '__main__':

    log_filepath = ('/media/julian/Data/projects/concepcion/'
                    'cryo_fib_pre_processing/2024-02-27_JC_C3/fiji_sift/raw-translation-fiji.txt')

    affines = sift_log_to_affine_stack_workflow(log_filepath, not_sequenced=True)
    affines_seq = sift_log_to_affine_stack_workflow(log_filepath)

    print(len(affines))
    # print(affines_seq['C', :])
    # print(affines.get_sequenced_stack()['C', :])
