
import numpy as np

if not hasattr(np, "float128"):
    np.float128 = np.longdouble


def load_affine_stack_from_multiple_files(filepaths, sequence_stack=False):

    stack = AffineStack(filepath=filepaths[0])
    if sequence_stack and not stack.is_sequenced:
        stack = stack.get_sequenced_stack()
    for filepath in filepaths[1:]:
        if not sequence_stack:
            stack.append(AffineStack(filepath=filepath))
        else:
            stack_ = AffineStack(filepath=filepath)
            if not stack_.is_sequenced:
                stack_ = stack_.get_sequenced_stack()
            stack.append(stack_.new_stack_with_same_meta(stack_ * stack[-1]))

    return stack


class AffineStack:

    def __init__(
            self,
            stack=None,
            filepath=None,
            is_sequenced=None,
            pivot=None
    ):

        self._filepath = None
        self._stack = None
        self._ndim = None
        self.is_sequenced = None
        self._it = 0
        self._pivot = None
        self._meta = dict()
        if stack is not None:
            assert filepath is None
            self.set_from_stack(stack, is_sequenced=is_sequenced, pivot=pivot)
            return
        if filepath is not None:
            assert stack is None
            self.set_from_file(filepath, is_sequenced=is_sequenced, pivot=pivot)
            return
        self.is_sequenced = is_sequenced
        self.set_pivot(pivot)

    def _validate_stack(self):
        pivots = [m.get_pivot() for m in self[:]]
        if not np.array((x == pivots[0]).all() for x in pivots).all():
            print(f'All pivots of stack must be equal!')
            return False
        if not (self._pivot == pivots[0]).all():
            print('Pivots in stack must match stack pivot')
            return False
        if len(self._pivot) != self._ndim:
            print(f'Pivot shape must match number of dimensions! pivot={self._pivot}; ndim={self._ndim}')
            return False
        ndims = np.array([m.get_ndim() for m in self[:]])
        if not np.array(x == ndims for x in ndims).all():
            print('All ndims of stack must be equal!')
            return False
        if self._ndim != ndims[0]:
            print(f'ndims in stack must match stack ndim! {ndims[0]} != {self._ndim}')
            return False
        return True

    def set_from_stack(self, stack, is_sequenced=None, pivot=None):
        if not isinstance(stack[0], AffineMatrix):
            stack = [AffineMatrix(parameters=parameters, pivot=pivot) for parameters in stack]
        if isinstance(stack, np.ndarray):
            stack = stack.tolist()
        self._stack = stack
        self._set_ndim()
        self.set_pivot(pivot)
        if self._validate_stack():
            self.is_sequenced = is_sequenced
            return
        raise RuntimeError('Validation of stack failed!')

    def update_stack(self, stack):
        self.set_from_stack(stack, is_sequenced=self.is_sequenced, pivot=self.get_pivot())

    def set_from_file(self, filepath, is_sequenced=None, pivot=None):
        import json
        with open(filepath, mode='r') as f:
            stack_data = json.load(f)

        stack = stack_data['transforms']
        if is_sequenced is None:
            is_sequenced = stack_data['sequenced']
        if pivot is None:
            pivot = stack_data['pivot']

        if 'meta' in stack_data:
            self._meta = stack_data['meta']

        self.set_from_stack(stack, is_sequenced=is_sequenced, pivot=pivot)

    def to_file(self, filepath):
        import json
        out_data = dict(
            transforms=self['C', :].astype('float64').tolist(),
            sequenced=self.is_sequenced,
            pivot=self.get_pivot().tolist(),
            meta=self._meta
        )
        try:
            with open(filepath, 'w') as f:
                json.dump(out_data, f, indent=2)
        except TypeError:
            print(out_data)
            raise

    def set_pivot(self, pivot=None):
        if pivot is None and self._stack is not None:
            pivots = [m.get_pivot() for m in self[:]]
            assert np.array((x == np.array(pivots[0])).all() for x in pivots).all(), 'Not all pivots are equal!'
            self._pivot = np.array(pivots[0])
            return
        if pivot is None and self._stack is None:
            self._pivot = np.array([0.] * self._ndim)
            return
        if pivot is not None and self._stack is not None:
            assert len(pivot) == self._ndim
            self._pivot = np.array(pivot, dtype=float)
            return
        if pivot is not None and self._stack is None:
            self._pivot = np.array(pivot, dtype=float)
            self._ndim = len(pivot)

    def _set_ndim(self):
        ndims = np.array([m.get_ndim() for m in self[:]])
        if (ndims == 2).all():
            self._ndim = 2
            return
        if (ndims == 3).all():
            self._ndim = 3

    def get_pivot(self):
        return self._pivot

    def _append_meta(self, other):
        for k, v in self._meta.items():
            if k in other._meta:
                if type(self._meta[k]) == list:
                    assert type(other._meta[k]) == list
                    self._meta[k].extend(other._meta[k])

    def append(self, other):
        if isinstance(other, AffineStack):
            assert self.is_sequenced == other.is_sequenced
            assert (self.get_pivot() == other.get_pivot()).all(), f'{self.get_pivot()} != {other.get_pivot()}'
            self.set_from_stack(self[:] + other[:], is_sequenced=self.is_sequenced, pivot=self.get_pivot())
            self._append_meta(other)
            return
        if isinstance(other, AffineMatrix):
            if (self.get_pivot() != other.get_pivot()).any():
                if len(self) == 0:
                    self.set_pivot(other.get_pivot())
            assert (self.get_pivot() == other.get_pivot()).all(), f'{self.get_pivot()} != {other.get_pivot()}'
            self.set_from_stack(self[:] + [other], is_sequenced=self.is_sequenced, pivot=self.get_pivot())

    def get_ndim(self):
        return self._ndim

    def __getitem__(self, item):
        """
        :param item:
        as[i]       -> AffineMatrix object
        as['Ms', i] -> [[a, b, ty], [c, d, tx], [0, 0, 1]]
        as['M', i]  -> [[a, b, ty], [c, d, tx]]
        as['Cs', i] -> [a, b, ty, c, d, tx, 0, 0, 1]
        as['C', i]  -> [a, b, ty, c, d, tx]
        as[:]       -> AffineStack[AffineMatrixA, AffineMatrixB, ...]
        as['Ms', :] -> [[[a, b, ty], [c, d, tx], [0, 0, 1]], ...]
        :return:
        """
        if isinstance(item, int):
            return self._stack[item]
        if isinstance(item, tuple):
            assert isinstance(item[0], str)
            items = self._stack[item[1]]
            if isinstance(items, AffineMatrix):
                return items.get_matrix(order=item[0])
            return np.array([x.get_matrix(order=item[0]) for x in items])
        if isinstance(item, slice):
            if self._stack is not None:
                return self._stack[item]
            else:
                return []
        raise ValueError('Invalid indexing!')

    def __setitem__(self, key, value):
        self._stack[key] = value

    def __neg__(self):
        return AffineStack(
            stack=[-x for x in self[:]],
            is_sequenced=self.is_sequenced
        )

    def __len__(self):
        if self._stack is None:
            return 0
        return len(self._stack)

    def __iter__(self):
        self._it = 0
        return self

    def __next__(self):
        if self._it < len(self):
            x = self[self._it]
            self._it += 1
            return x
        else:
            raise StopIteration

    def __mul__(self, other):
        if isinstance(other, AffineStack):
            return AffineStack(
                stack=[self[idx] * other[idx] for idx in range(len(self))],
                is_sequenced=self.is_sequenced
            )
        if isinstance(other, AffineMatrix):
            return AffineStack(
                stack=[self[idx] * other for idx in range(len(self))],
                is_sequenced=self.is_sequenced
            )
        raise ValueError(f'Invalid input for argument other: {type(other)}')

    def new_stack_with_same_meta(self, new_stack):
        ns = AffineStack(stack=new_stack, is_sequenced=self.is_sequenced, pivot=self.get_pivot())
        ns._meta = self._meta
        return ns

    def copy(self):
        return self.new_stack_with_same_meta(self._stack.copy())

    def get_smoothed_stack(self, sigma, mode='reflect'):
        from scipy.ndimage import gaussian_filter1d
        dtype = self[0].get_dtype()
        return self.new_stack_with_same_meta(gaussian_filter1d(self['C', :].astype('float64'), sigma, axis=0, mode=mode).astype(dtype))

    def get_median_smoothed_stack(self, radius):
        from scipy.ndimage import median_filter

        dtype = self[0].get_dtype()

        # window size only along axis 0
        size = [1] * self['C', :].ndim
        size[0] = 2 * radius + 1

        smoothed = median_filter(
            self['C', :].astype('float64'),
            size=size,
            mode='nearest'
        )

        return self.new_stack_with_same_meta(smoothed.astype(dtype))

    def get_sequenced_stack(self):

        assert not self.is_sequenced, 'A sequenced stack cannot be sequenced again!'

        """
        # This is the original version which - in principle - does the same as new version below.
        # The difference is that in this original version the sequence of dot product operations becomes very long for 
        # large image stacks such that small errors accumulate.
        # The new version runs pairs of the sequence in parallel thus avoiding the issue.
        
        stack = []
        for idx, matrix in enumerate(self[:]):
            if idx == 0:
                stack.append(self[0])
                continue
            stack.append(stack[idx - 1] * self[idx])

        out_stack = self.new_stack_with_same_meta(stack)
        out_stack.is_sequenced = True
        return out_stack
        
        """

        def group_stack(stack):
            return [
                [stack[idx], stack[idx + 1]]
                if idx < len(stack) - 1
                else [stack[idx]]
                for idx in range(0, len(stack), 2)
            ]

        def merge_groups(a, b):
            return [a[-1] * x for x in b]

        new_stack = np.array(self[:])[:, None].tolist()

        iteration = 0
        while len(new_stack) > 1:
            this_stack = group_stack(new_stack)
            new_stack = []
            for group in this_stack:
                if len(group) > 1:
                    group[1] = merge_groups(group[0], group[1])
                    new_stack.append(group[0] + group[1])
                else:
                    new_stack.append(group[0])

            iteration += 1

        new_stack = self.new_stack_with_same_meta(np.array(new_stack).flatten())
        new_stack.is_sequenced = True
        return new_stack

    def get_not_sequenced_stack(self):

        new_stack = []
        for idx in range(1, len(self)):
            new_stack.append(self[idx] * -self[idx - 1])
        new_stack = self.new_stack_with_same_meta(np.array(new_stack).flatten())
        new_stack.is_sequenced = False
        return new_stack

    @staticmethod
    def _z_interpolate(stack, scale):
        stack = np.array(stack)
        dtype = stack.dtype
        stack = stack.astype('float64')

        if scale > 1 or 1/scale - int(1/scale) != 0:
            # z-interpolation to extend the stack
            from scipy.ndimage import zoom
            stack = zoom(
                stack, (scale, 1.),
                order=1, grid_mode=True, mode='grid-constant'
            )
        else:
            stack = [
                stack[idx]
                for idx in range(0, len(stack), int(1/scale))
            ]
        return np.array(stack, dtype=dtype)

    def get_scaled(self, scale):

        assert self.is_sequenced, 'Scaling only works for sequenced stacks!'

        scaled_stack = [item.get_scaled(scale).get_matrix('C') for item in self]
        scaled_stack = self._z_interpolate(scaled_stack, scale)

        return self.new_stack_with_same_meta(scaled_stack)

    def get_interpolated(self, scale):
        stack = [item.get_matrix('C') for item in self]
        stack = self._z_interpolate(stack, scale)
        return self.new_stack_with_same_meta(stack)

    def apply_z_step(self):

        assert 'z_step' in self._meta
        assert self.is_sequenced

        z_step = self.get_meta('z_step')
        if z_step == 1:
            return self

        stack = np.array(self['C', :])

        from scipy.interpolate import CubicSpline

        new_stack = []
        for seq in stack.swapaxes(0, 1):

            x = np.arange(len(seq))
            y = seq
            interpolator = CubicSpline(x, y, extrapolate=True, bc_type='natural')

            y_ = np.arange(0, len(seq), 1 / z_step)  # [:len(self)]
            new_stack.append(interpolator(y_))

        new_stack = np.swapaxes(new_stack, 0, 1)
        # return AffineStack(stack=new_stack, is_sequenced=True, pivot=self._pivot)
        new_stack = self.new_stack_with_same_meta(new_stack)
        new_stack.set_meta('z_step', 1)
        print(self.is_sequenced)
        return new_stack

    def set_meta(self, name=None, data=None):
        if name is None:
            assert isinstance(data, dict)
            self._meta = data
            return
        if type(data) == np.ndarray:
            data = data.tolist()
        self._meta[name] = data

    def get_meta(self, name=None):
        if name is not None:
            return self._meta[name]
        return self._meta

    def exists_meta(self, name):
        if name in self._meta.keys():
            return True
        return False

    def get_translations(self):
        return [x.get_translation() for x in self]

    def set_translations(self, translations):
        assert len(translations) == len(self)
        for idx, m in enumerate(self):
            m.set_translation(translations[idx])

    def add_to_translations(self, values):
        translations = np.array(self.get_translations())
        self.set_translations(translations + np.array(values))

    def get_substack(self, sl: slice):
        substack = self.new_stack_with_same_meta(self[sl])
        if self.exists_meta('bounds'):
            bounds = self.get_meta('bounds')
            substack.set_meta('bounds', bounds[sl])
        if self.exists_meta('stack_shape'):
            stack_shape = self.get_meta('stack_shape')
            substack.set_meta('stack_shape', [sl.stop - sl.start, stack_shape[1], stack_shape[2]])
        return substack
    
    def large_offsets_to_zero(self, max_offset_distance):
        """
        For translation transforms, set large offsets to zero to limit the search space for the translation.
        This is useful when the translation is expected to be small and large offsets are likely due to noise or artifacts.
        Note that this does not check or change the affine part of the transform.

        :param max_offset_distance: Maximum allowed offset distance in pixels. Offsets larger than this will be set to zero.
        """
        assert self.get_ndim() == 2 or self.get_ndim() == 3, "This method only applies to 2D or 3D affine transforms."
        assert not self.is_sequenced, "This method only applies to non-sequenced affine stacks."
        translations = self.get_translations()
        translations = np.array(translations)
        print(f'translations = {translations}')
        distances = np.linalg.norm(translations, axis=1)
        translations[distances > max_offset_distance] = 0
        print(f'translations = {translations}')
        self.set_translations(translations)


class AffineMatrix:

    def __init__(
            self,
            parameters=None,
            elastix_parameters=None,
            filepath=None,
            translation=None,
            rotation=None,
            pivot=None
    ):
        """

        :param parameters:
        :param elastix_parameters: Supply Elastix parameters in the format [transform: str, transform_parameters: list]
            or as SimpleITK.SimpleITK.ParameterMap
        :param filepath:
        :param pivot:
        """
        self._parameters = None
        self._ndim = None
        self._pivot = None
        if parameters is not None:
            self.set_from_parameters(parameters, pivot=pivot)
        if elastix_parameters is not None:
            self.set_from_elastix(elastix_parameters, pivot=pivot)
        if filepath is not None:
            self.set_from_file(filepath)
        if translation is not None:
            self.set_from_translation(translation, pivot=pivot)
        if rotation is not None:
            self.set_from_rotation(rotation, pivot=pivot)

    def _validate_parameters(self, parameters):
        """
        Accepts any layout of 2D or 3D affine matrices, including flattened homogeneous:
          - 2D flat: 6 or 9 elements
          - 2D matrix: 2x3 or 3x3
          - 3D flat: 12 or 16 elements
          - 3D matrix: 3x4 or 4x4
        Returns flattened layout:
          - 2D -> 6 elements
          - 3D -> 12 elements
        """
        arr = np.array(parameters, dtype=float).squeeze()
        normalized = None
        ndim = None

        # ---- Already flattened ----
        if arr.ndim == 1:
            if len(arr) == 6:
                ndim = 2
                normalized = arr
            elif len(arr) == 9:  # 2D homogeneous flattened
                ndim = 2
                normalized = np.array([
                    arr[0], arr[1], arr[2],
                    arr[3], arr[4], arr[5]
                ])
            elif len(arr) == 12:
                ndim = 3
                normalized = arr
            elif len(arr) == 16:  # 3D homogeneous flattened
                ndim = 3
                normalized = np.array([
                    arr[0], arr[1], arr[2], arr[3],
                    arr[4], arr[5], arr[6], arr[7],
                    arr[8], arr[9], arr[10], arr[11]
                ])

        # ---- 2D / 3D matrix layouts ----
        elif arr.ndim == 2:
            rows, cols = arr.shape

            # 2D affine 2x3
            if rows == 2 and cols == 3:
                ndim = 2
                normalized = arr.flatten()

            # 2D homogeneous 3x3
            elif rows == 3 and cols == 3:
                ndim = 2
                normalized = arr[:2, :].flatten()

            # 3D affine 3x4
            elif rows == 3 and cols == 4:
                ndim = 3
                normalized = arr.flatten()

            # 3D homogeneous 4x4
            elif rows == 4 and cols == 4:
                ndim = 3
                normalized = arr[:3, :].flatten()

        if normalized is None:
            raise ValueError(
                f"Invalid affine layout. Expected 2D (6 or 3x3) or 3D (12 or 4x4), got shape {arr.shape}"
            )

        self._ndim = ndim
        return normalized

    def set_from_parameters(self, parameters, pivot=None):
        parameters = self._validate_parameters(parameters)
        self._parameters = np.array(parameters, dtype=np.float128)
        self.set_pivot(pivot)
        return

    def set_from_translation(self, translation, pivot=None):
        self.set_from_parameters(
            np.concatenate([np.eye(len(translation)), np.array([translation]).T], axis=1).flatten(),
            pivot=pivot
        )

    def set_from_rotation(self, rotation, pivot=None):
        """
        Set affine parameters from a rotation.
        Acceptable inputs:
          - 2D: single angle (float, radians)
          - 3D:
              * 3×3 rotation matrix
              * iterable of 3 Euler angles in radians, assumed (rx, ry, rz) intrinsic
        """
        rot = np.array(rotation, dtype=float)

        # ---------- 2D ROTATION ----------
        if rot.ndim == 0 or (rot.ndim == 1 and rot.size == 1):
            # Single angle
            theta = float(rot)
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            T = np.zeros(2)
            affine = np.hstack([R, T.reshape(2, 1)])  # 2×3
            self.set_from_parameters(affine, pivot=pivot)
            return

        # ---------- 3D ROTATION MATRIX ----------
        if rot.ndim == 2 and rot.shape == (3, 3):
            R = rot
            T = np.zeros(3)
            affine = np.hstack([R, T.reshape(3, 1)])  # 3×4
            self.set_from_parameters(affine, pivot=pivot)
            return

        # ---------- 3D EULER ANGLES ----------
        if rot.ndim == 1 and rot.size == 3:
            rx, ry, rz = rot.astype(float)

            # Rotation matrices (intrinsic rotations)
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)]
            ])
            Ry = np.array([
                [np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)]
            ])
            Rz = np.array([
                [np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz), np.cos(rz), 0],
                [0, 0, 1]
            ])

            # Final R = Rz * Ry * Rx
            R = Rz @ Ry @ Rx
            T = np.zeros(3)
            affine = np.hstack([R, T.reshape(3, 1)])
            self.set_from_parameters(affine, pivot=pivot)
            return

        raise ValueError(
            "Rotation must be: angle (float), 3×3 matrix, or Euler angles (3-vector). "
            f"Got shape {rot.shape}"
        )

    def update_parameters(self, parameters):
        self.set_from_parameters(parameters, pivot=self.get_pivot())

    def set_from_elastix(self, parameters, pivot=None):
        import SimpleITK as sitk
        if type(parameters) == sitk.SimpleITK.ParameterMap:
            parameters = [parameters['Transform'][0], parameters['TransformParameters']]
            if parameters[0] == 'TranslationTransform':
                parameters[0] = 'translation'
        assert isinstance(parameters[0], str), \
            'Elastix parameters must be in the format: ["transform", [parameters, ...]]'
        assert parameters[0] in ['translation', 'TranslationTransform', 'rigid', 'EulerTransform', 'SimilarityTransform', 'affine', 'AffineTransform'], f'invalid transform: {parameters[0]}'
        from ..library.elastix import elastix_to_c
        parameters = elastix_to_c(*parameters)
        self.set_from_parameters(parameters, pivot=pivot)

    def get_matrix(self, order='C'):

        if order[0] == 'C':
            if len(order) == 2 and order[1] == 's':
                return np.concatenate((self._parameters.copy(), [0.] * self._ndim + [1.]), axis=0)
            return self._parameters.copy()
        if order[0] == 'M':
            out_matrix = np.reshape(self._parameters.copy(), (self._ndim, self._ndim + 1), order='C')
            if len(order) == 2 and order[1] == 's':
                return np.concatenate((out_matrix, [[0.] * self._ndim + [1.]]), axis=0)
            return out_matrix

    def set_from_file(self, filepath):
        from squirrel.library.io import get_filetype
        filetype = get_filetype(filepath)

        if filetype == 'json':
            import json
            with open(filepath, mode='r') as f:
                matrix_data = json.load(f)
                self.set_from_parameters(
                    parameters=matrix_data['transform'],
                    pivot=matrix_data['pivot'] if 'pivot' in matrix_data else None
                )
            return

        if filetype == 'csv':
            from numpy import genfromtxt
            self.set_from_parameters(parameters=genfromtxt(filepath, delimiter=',').flatten())
            return

        raise ValueError(f'Invalid filetype: {filetype}')

    def set_pivot(self, pivot):
        self._pivot = np.array(pivot, dtype=float) if pivot is not None else np.array([0.] * self._ndim)
        assert len(self._pivot) == self._ndim

    def get_pivot(self):
        return self._pivot

    def to_file(self, filepath):
        import json
        out_data = dict(
            transform=self.get_matrix(order='C').astype('float64').tolist(),
            pivot=self._pivot.tolist()
        )
        with open(filepath, mode='w') as f:
            json.dump(out_data, f, indent=2)

    def _ms_to_c(self, matrix):
        if self._ndim == 2:
            assert (matrix[2] == [0., 0., 1.]).all()
            return matrix[:2].flatten()
        if self._ndim == 3:
            assert (matrix[3] == [0., 0., 0., 1]).all()
            return matrix[:3].flatten()

    def get_ndim(self):
        return self._ndim

    def inverse(self):
        inv = np.linalg.inv(self.get_matrix(order='Ms').astype(np.float64))
        inv = self._ms_to_c(inv)
        return AffineMatrix(parameters=inv, pivot=self.get_pivot())

    def dot(self, other):
        assert type(other) == AffineMatrix
        assert np.all(other.get_pivot() == self.get_pivot())
        result = np.dot(self.get_matrix(order='Ms').astype(np.float128), other.get_matrix(order='Ms').astype(np.float128))
        result = self._ms_to_c(result)
        return AffineMatrix(parameters=result, pivot=self.get_pivot())

    def __neg__(self):
        return self.inverse()

    def __mul__(self, other):
        return self.dot(other)

    def copy(self):
        return AffineMatrix(self.get_matrix(), pivot=self.get_pivot())

    def get_translation(self):
        return self.get_matrix('M')[:self._ndim, self._ndim]

    def set_translation(self, translation):
        assert len(translation) == self._ndim
        matrix = self.get_matrix('M')
        matrix[:self._ndim, self._ndim] = translation
        self.set_from_parameters(matrix.flatten(), pivot=self.get_pivot())

    def get_scaled(self, scale):
        matrix = self.copy()
        matrix.set_translation(matrix.get_translation() * scale)
        pivot_matrix = AffineMatrix(translation=matrix.get_pivot(), pivot=matrix.get_pivot())
        matrix = matrix * pivot_matrix
        pivot_matrix.set_translation(pivot_matrix.get_translation() * scale)
        return (-pivot_matrix) * matrix

    def decompose(self):
        from transforms3d.affines import decompose
        from squirrel.library.transformation import (
            setup_translation_matrix,
            setup_scale_matrix,
            setup_shear_matrix
        )
        t, r, z, s = decompose(self.get_matrix('Ms').astype('float64'))
        r_ = np.zeros([self._ndim, self._ndim + 1], dtype=float)
        r_[:r.shape[0], :r.shape[1]] = r
        return (
            AffineMatrix(setup_translation_matrix(t, ndim=self.get_ndim()).flatten()),
            AffineMatrix(r_.flatten()),
            AffineMatrix(setup_scale_matrix(z, ndim=self.get_ndim()).flatten()),
            AffineMatrix(setup_shear_matrix(s, ndim=self.get_ndim()).flatten())
        )

    def shift_pivot_to_origin(self):
        matrix = self.get_matrix('Ms')
        pivot = self.get_pivot()
        offset = pivot - np.dot(matrix[:self._ndim, :self._ndim], pivot)
        pivot_matrix = np.eye(self._ndim + 1)
        pivot_matrix[:self._ndim, self._ndim] = offset
        matrix = np.dot(pivot_matrix, matrix)[:self._ndim]
        self.set_from_parameters(matrix.flatten(), pivot=[0.] * self._ndim)

    def get_dtype(self):
        return self._parameters.dtype

    def to_elastix_affine(self, shape=None, return_parameter_map=False):

        from squirrel.library.elastix import c_to_elastix
        affine_elastix = c_to_elastix(self.get_matrix().astype('float64'))

        if not return_parameter_map:
            return affine_elastix

        num_params = len(affine_elastix)

        import SimpleITK as sitk
        affine_params = sitk.ParameterMap()
        affine_params['TransformParameters'] = [str(x) for x in affine_elastix]
        affine_params['NumberOfParameters'] = [str(num_params)]
        affine_params['CenterOfRotationPoint'] = [str(x) for x in self.get_pivot()[::-1]]
        affine_params['Transform'] = ['AffineTransform']
        affine_params['Spacing'] = ['1'] * self._ndim
        if shape is not None:
            affine_params['Size'] = [str(x) for x in shape[::-1]]
        affine_params['Index'] = ['0'] * self._ndim
        affine_params['Origin'] = ['0'] * self._ndim
        affine_params['Direction'] = [str(x) for x in np.eye(self._ndim).flatten()]
        affine_params['UseDirectionCosines'] = ['true']

        return affine_params

    def _expand_pivot(self, axis):
        """Helper: expand 2D pivot into 3D with zero inserted at the given axis."""
        px, py = self.get_pivot()[:2]
        if axis == 0:
            return [0, px, py]
        elif axis == 1:
            return [px, 0, py]
        else:
            return [px, py, 0]

    def return_3d(self, axis=2):
        """
        Converts a 2D affine transform to a 3D transform by embedding
        it in the plane orthogonal to the specified axis.

        Parameters
        ----------
        axis : int, optional
            Axis along which to introduce the third dimension (default: 2).
            - axis = 0 → embed in YZ plane
            - axis = 1 → embed in XZ plane
            - axis = 2 → embed in XY plane

        Returns
        -------
        AffineMatrix
            New 3D AffineMatrix object (flattened 12-element form).
        """
        if self._ndim == 3:
            return self.copy()

        if self._ndim != 2:
            raise ValueError("return_3d() only works on 2D affine transforms.")

        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2.")

        a00, a01, t0, a10, a11, t1 = self._parameters

        # Start with an identity 3D transform
        M = np.eye(3, 4)

        if axis == 2:
            # Default XY plane
            M[0, 0], M[0, 1], M[0, 3] = a00, a01, t0
            M[1, 0], M[1, 1], M[1, 3] = a10, a11, t1
        elif axis == 1:
            # XZ plane (omit Y)
            M[0, 0], M[0, 2], M[0, 3] = a00, a01, t0
            M[2, 0], M[2, 2], M[2, 3] = a10, a11, t1
        elif axis == 0:
            # YZ plane (omit X)
            M[1, 1], M[1, 2], M[1, 3] = a00, a01, t0
            M[2, 1], M[2, 2], M[2, 3] = a10, a11, t1

        affine_3d = AffineMatrix(
            parameters=M.flatten(),
            pivot=self._expand_pivot(axis)
        )
        return affine_3d


if __name__ == '__main__':

    # TODO: the stuff here should be implemented as unittests

    if True:
        astack = AffineStack(
            stack=[
                [1, 0, 1, 0, 1, 1],
                [1, 0, 2, 0, 1, 2],
                [1, 0, 3, 0, 1, 3],
            ],
            is_sequenced=False
        )

        print(astack['Ms', :])
        astack.large_offsets_to_zero(max_offset_distance=1.1)
        print(astack['Ms', :])

    if False:
        print('Testing the stack object -----------------')
        stk = AffineStack(
            stack=[[1.1, 0., 5, 0., 1.3, 2], [0.8, 0., -3, 0., 0.75, 1.5]],
            is_sequenced=False
        )
        # print(f'stk.get_ndim = {stk.get_ndim()}')
        # print(f'stk[0] = {stk[0]}')
        # print(f'stk[0].get_matrix() = {stk[0].get_matrix()}')
        # print(f'stk[1].get_matrix() = {stk[1].get_matrix()}')
        # print(f'stk["Ms", 0] = {stk["Ms", 0]}')
        # print(f'stk[:] = {stk[:]}')
        # print(f'stk["Ms", :] = {stk["Ms", :]}')
        # print(f'\nStack IO ---------------------')
        stk.set_meta('test', np.array([1, 2, 4]))
        fp = '/media/julian/Data/tmp/affine_stack_test.json'
        stk.to_file(fp)
        stk_loaded = AffineStack(filepath=fp)
        print(f'stk_loaded["Ms", :] = {stk_loaded["Ms", :]}')
        print(f'stk_loaded.is_sequenced = {stk_loaded.is_sequenced}')
        print(f'\nMathematical operations ---------------------')
        print(f'(-stk)["Ms", :] = {(-stk)["Ms", :]}')
        print(f'(stk * stk)["Ms", :] = {(stk * stk)["Ms", :]}')
        print(f'(stk * -stk)["Ms", :] = {(stk * -stk)["Ms", :]}')
        stk.append(-stk_loaded)
        print(f'Appended: stk["C", :] = {stk["C", :]}')
        # print(f'stk.smooth(2)["Ms", :] = {stk.get_smoothed_stack(2)["Ms", :]}')
        # print(f'before: stk["Ms", :] = {stk["C", :]}')
        # print(f'sequenced: stk["Ms", :] = {stk.get_sequenced_stack()["C", :]}')
        print(f'len(stk) = {len(stk)}')
        stk = stk.get_sequenced_stack()
        print(f'len(stk) = {len(stk)}')
        # stk.is_sequenced = True
        print(f'scaled: {stk.get_scaled(2)["C", :]}')
        print(f'scaled: {stk.get_scaled(0.5)["C", :]}')
        print(f'translations: {stk.get_translations()}')
        stk.add_to_translations([1, 2])
        print(f'translations: {stk.get_translations()}')
        fp2 = '/media/julian/Data/tmp/affine_stack_test2.json'
        stk.to_file(fp2)
        stk2 = load_affine_stack_from_multiple_files([fp, fp2], sequence_stack=True)
        print(stk2['C', :])

    if False:
        print('Testing the matrix object')
        am = AffineMatrix(parameters=[1.1, 0., 5, 0., 1.3, 2])
        # print(am.get_matrix(order='C'))
        # print(am.get_matrix(order='Cs'))
        # print(am.get_matrix(order='M'))
        # print(am.get_matrix(order='Ms'))
        # print((-am).get_matrix(order='Ms'))
        # print((am * am).get_matrix(order='Ms'))
        # print((am * -am).get_matrix(order='Ms'))
        # print((-am * am).get_matrix(order='Ms'))
        # print(am.get_matrix(order='Ms'))
        # print('\nTesting IO')
        # fp = '/media/julian/Data/tmp/affine_matrix_test.json'
        # am.to_file(fp)
        # am_loaded = AffineMatrix()
        # am_loaded.set_from_file(fp)
        # print(am_loaded.get_matrix(order='Ms'))
        # am3d = AffineMatrix(parameters=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], pivot=[100, 0, 0])
        # am3d.to_file('/media/julian/Data/tmp/affine_matrix_test_3d.json')

        print(f'non-scaled: {am.get_matrix("Ms")}')
        print(f'scaled: {am.get_scaled(2).get_matrix("Ms")}')
        print(f'non-scaled: {am.get_matrix("Ms")}')
        # from squirrel.library.transformation import scale_affine_matrix
        # print(scale_affine_matrix(am.get_matrix('Ms'), 2, [0., 0.]))

        print(f'decomposed:')
        decomp = am.decompose()
        for d in decomp:
            print(d.get_matrix())

