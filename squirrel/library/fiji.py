
import numpy as np


def sift_log_line_to_affine(line, pivot):
    import re

    pattern = r'[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?'
    matrix = np.array([float(x) for x in re.findall(pattern, line)])

    from squirrel.library.affine_matrices import AffineMatrix
    # The fiji matrix has inverted x and y axes
    m = AffineMatrix([matrix[4], matrix[3], matrix[5], matrix[1], matrix[0], matrix[2]], pivot=pivot).inverse()
    # m.set_translation([-matrix[5], -matrix[2]])
    return m


def sift_log_to_affine_stack(
        log_lines,
        pivot=None
):

    from squirrel.library.affine_matrices import AffineStack

    pivot = [0., 0.] if pivot is None else pivot

    transforms = AffineStack(sequenced=False)
    for line in log_lines:
        if line.startswith('Transformation Matrix:'):
            transforms.append(sift_log_line_to_affine(line, pivot))

    return transforms


if __name__ == '__main__':

    log_filepath = ('/media/julian/Data/projects/concepcion/'
                    'cryo_fib_pre_processing/2024-02-27_JC_C3/fiji_sift/raw-translation-fiji.txt')
    f = open(log_filepath, mode='r')
    l = f.readlines()

    affines = sift_log_to_affine_stack(l)

    print(affines['C', :])
