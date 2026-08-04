
from __future__ import annotations
import numpy as np
from collections.abc import Iterable, Sequence
from copy import deepcopy


if not hasattr(np, "float128"):
    np.float128 = np.longdouble


def _jsonify(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {key: _jsonify(value) for key, value in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_jsonify(value) for value in obj]

    return obj
    


class AffineStack:
    """
    Represents an ordered collection of affine transformations.

    Each item corresponds to one image slice or stack position.

    The stack can contain either:

    - relative transforms:
      each matrix maps one slice relative to the previous slice;

    - sequenced transforms:
      each matrix maps a slice into a common reference coordinate system.
    """

    def __init__(
        self,
        matrices=None,
        sequenced=False,
        metadata=None,
    ):
        self._matrices = []
        self._ndim = None
        self._pivot = None
        self._dtype = None

        self._sequenced = self._validate_sequenced(sequenced)
        self._metadata = self._normalize_metadata(metadata)

        if matrices is not None:
            self.extend(matrices)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def identity(
        cls,
        length: int,
        ndim: int,
        pivot: np.typing.ArrayLike | None = None,
        sequenced: bool = False,
        metadata: dict | None = None,
        dtype: np.dtype | type = np.float64,
    ) -> AffineStack:

        if not isinstance(length, (int, np.integer)):
            raise TypeError("length must be an integer.")

        if length < 0:
            raise ValueError(f"length must be non-negative; received {length}.")

        stack = cls(sequenced=sequenced, metadata=metadata)

        for _ in range(length):
            stack.append(AffineMatrix.identity(ndim=ndim, pivot=pivot, dtype=dtype))

        return stack

    @classmethod
    def from_array(
        cls,
        array: np.typing.ArrayLike,
        pivot: np.typing.ArrayLike | None = None,
        sequenced: bool = False,
        metadata: dict | None = None,
    ) -> AffineStack:
        """
        Construct an AffineStack from an array of affine matrices.

        Accepted layouts include stacks of:

        - flat compact matrices;
        - flat homogeneous matrices;
        - compact matrices;
        - homogeneous matrices.

        Examples
        --------
        2D compact stack:

            shape (n, 2, 3)

        2D flat compact stack:

            shape (n, 6)

        3D homogeneous stack:

            shape (n, 4, 4)
        """
        stack_array = np.asarray(array)

        if not np.issubdtype(stack_array.dtype, np.number):
            raise TypeError("Affine stack values must be numeric.")

        if stack_array.ndim < 2:
            raise ValueError(f"An affine stack array must contain a leading stack dimension; received shape {stack_array.shape}.")

        matrices = [AffineMatrix.from_array(matrix, pivot=pivot) for matrix in stack_array]

        return cls(matrices=matrices, sequenced=sequenced, metadata=metadata)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def pivot(self):
        return None if self._pivot is None else self._pivot.copy()

    @property
    def dtype(self):
        return self._dtype

    @property
    def sequenced(self) -> bool:
        return self._sequenced

    @property
    def metadata(self) -> dict:
        return deepcopy(self._metadata)

    # ------------------------------------------------------------------
    # Array conversion
    # ------------------------------------------------------------------

    def as_array(self, homogeneous: bool = True, flatten: bool = False):
        if len(self) == 0:
            return np.empty((0,))
        return np.stack([matrix.as_array(homogeneous=homogeneous, flatten=flatten) for matrix in self])

    def as_compact(self):
        """
        Return the stack as compact affine matrices.

        2D: shape (n, 2, 3)
        3D: shape (n, 3, 4)
        """
        return self.as_array(homogeneous=False)

    def as_homogeneous(self):
        """
        Return the stack as homogeneous affine matrices.

        2D: shape (n, 3, 3)
        3D: shape (n, 4, 4)
        """
        return self.as_array(homogeneous=True)

    # ------------------------------------------------------------------
    # Stack conversion
    # ------------------------------------------------------------------

    def to_sequenced(self):
        """
        Return a sequenced affine stack.

        Relative transforms are accumulated such that each transform maps into
        the common reference frame of the first slice.
        """
        if self.sequenced:
            return self.copy()

        stack = AffineStack(sequenced=True, metadata=self.metadata)
        current = AffineMatrix.identity(ndim=self.ndim, pivot=self.pivot, dtype=self.dtype)

        for matrix in self:
            current = current @ matrix
            stack.append(current)

        return stack

    def to_relative(self):
        """
        Return a relative affine stack.
        
        Sequenced transforms are converted to pairwise relative transforms.
        """
        if not self.sequenced:
            return self.copy()

        stack = AffineStack(sequenced=False, metadata=self.metadata)
        previous = AffineMatrix.identity(ndim=self.ndim, pivot=self.pivot, dtype=self.dtype)

        for matrix in self:
            stack.append(previous.inverse() @ matrix)
            previous = matrix

        return stack

    # ------------------------------------------------------------------
    # Stack operations
    # ------------------------------------------------------------------

    def append(self, matrix: AffineMatrix | np.typing.ArrayLike) -> None:

        if not isinstance(matrix, AffineMatrix):
            matrix = AffineMatrix.from_array(matrix)

        matrix = matrix.copy()

        if len(self._matrices) == 0:
            self._ndim = matrix.ndim
            self._pivot = matrix.pivot
            self._dtype = matrix.dtype
        else:
            if matrix.ndim != self._ndim:
                raise ValueError(f"Expected a {self._ndim}D affine matrix, received {matrix.ndim}D.")
            if not np.allclose(matrix.pivot, self._pivot):
                raise ValueError("Affine matrix pivot does not match stack pivot.")

            self._dtype = np.result_type(self._dtype, matrix.dtype)

        self._matrices.append(matrix)

    def _extend_metadata(self, other) -> None:
        """
        Extend per-slice metadata shared by both stacks.
        """
        for name, current_value in self._metadata.items():
            if name not in other._metadata:
                continue

            other_value = other._metadata[name]

            if isinstance(current_value, list):
                if not isinstance(other_value, list):
                    raise TypeError(f"Metadata entry {name!r} must have matching types.")

                current_value.extend(deepcopy(other_value))
                continue

            if isinstance(current_value, np.ndarray):
                if not isinstance(other_value, np.ndarray):
                    other_value = np.asarray(other_value)

                self._metadata[name] = np.concatenate([current_value, other_value], axis=0)
                
    def extend(self, matrices) -> None:
        if not isinstance(matrices, AffineStack):
            for matrix in matrices:
                self.append(matrix)
            return

        if len(self) > 0 and self.sequenced != matrices.sequenced:
            raise ValueError("Cannot extend stacks with different sequencing states.")

        previous_length = len(self)

        for matrix in matrices:
            self.append(matrix)

        if previous_length == 0:
            if not self._metadata:
                self._metadata = deepcopy(matrices.metadata)
            return

        self._extend_metadata(matrices)

    def composed_with(self, other):
        """
        Compose every matrix with another matrix or a matching stack.
        """
        if isinstance(other, AffineMatrix):
            return AffineStack(
                matrices=[matrix @ other for matrix in self],
                sequenced=self.sequenced,
                metadata=self.metadata,
            )

        if isinstance(other, AffineStack):
            if len(self) != len(other):
                raise ValueError(
                    "Affine stacks must have the same length for pairwise composition; received {len(self)} and {len(other)}."
                )

            if self.sequenced != other.sequenced:
                raise ValueError("Affine stacks must have the same sequencing state for pairwise composition.")

            return AffineStack(
                matrices=[left @ right for left, right in zip(self, other)],
                sequenced=self.sequenced,
                metadata=self.metadata,
            )

        return NotImplemented

    def inverse(self):
        """
        Return a stack containing the inverse of every transform.
        """
        return AffineStack(
            matrices=[matrix.inverse() for matrix in self],
            sequenced=self.sequenced,
            metadata=self.metadata,
        )

    def with_pivot(self, pivot):
        """
        Return a new stack with the pivot of every transform replaced.
        """
        if len(self) == 0:
            return self.copy()

        pivot = AffineMatrix._normalize_pivot(pivot, ndim=self.ndim)

        return AffineStack(
            matrices=[matrix.with_pivot(pivot) for matrix in self],
            sequenced=self.sequenced,
            metadata=self.metadata,
        )

    def with_translations(self, translations):
        """
        Return a new stack with replaced translation vectors.
        """
        translations = np.asarray(translations)
        expected_shape = (len(self), self.ndim)

        if translations.shape != expected_shape:
            raise ValueError(f"translations must have shape {expected_shape}; received {translations.shape}.")
        if not np.issubdtype(translations.dtype, np.number):
            raise TypeError("Translation values must be numeric.")
        if not np.isfinite(translations).all():
            raise ValueError("Translation values must all be finite.")

        return AffineStack(
            matrices=[matrix.with_translation(translation) for matrix, translation in zip(self, translations)],
            sequenced=self.sequenced,
            metadata=self.metadata,
        )

    def add_to_translations(self, offsets):
        """
        Return a new stack with offsets added to its translations.

        ``offsets`` may be either:
        - one vector with shape ``(ndim,)``;
        - one vector per transform with shape ``(len(self), ndim)``.
        """
        offsets = np.asarray(offsets)

        if offsets.shape == (self.ndim,):
            offsets = np.broadcast_to(offsets, (len(self), self.ndim))
        elif offsets.shape != (len(self), self.ndim):
            raise ValueError(
                f"offsets must have shape ({self.ndim},) or ({len(self)}, {self.ndim}); received {offsets.shape}."
            )

        translations = np.stack([matrix.translation for matrix in self])
        return self.with_translations(translations + offsets)

    def get_substack(self, selection):
        """
        Return a substack and slice matching per-transform metadata.
        """
        if not isinstance(selection, slice):
            raise TypeError("selection must be a slice.")

        indices = list(range(len(self)))[selection]
        metadata = self.metadata

        for name, value in list(metadata.items()):
            if isinstance(value, list) and len(value) == len(self):
                metadata[name] = value[selection]

            elif (isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == len(self)):
                metadata[name] = value[selection]

        if "stack_shape" in metadata:
            stack_shape = list(metadata["stack_shape"])

            if len(stack_shape) > 0:
                stack_shape[0] = len(indices)
                metadata["stack_shape"] = stack_shape

        return AffineStack(
            matrices=self._matrices[selection],
            sequenced=self.sequenced,
            metadata=metadata,
        )

    def apply_z_step(self, max_length=None):
        """
        Expand a sequenced stack according to the metadata entry ``z_step``.

        The affine parameters are interpolated along the stack axis using
        natural cubic splines. The returned stack has ``z_step`` metadata set
        to 1.
        """
        if not self.has_metadata("z_step"):
            raise ValueError("Cannot apply z-step because metadata contains no 'z_step' entry.")
        if not self.sequenced:
            raise ValueError("Applying z-step requires a sequenced AffineStack.")

        z_step = float(self.get_metadata("z_step"))

        if not np.isfinite(z_step) or z_step <= 0:
            raise ValueError("Metadata entry 'z_step' must be a finite value greater than zero.")

        if z_step == 1:
            return self.copy()

        if len(self) < 2:
            raise ValueError("Applying z-step interpolation requires at least two transforms.")

        parameters = self.as_array(homogeneous=False, flatten=True).astype(np.float64)
        source_positions = np.arange(len(self), dtype=float)
        target_positions = np.arange(0, len(self), 1.0 / z_step)

        from scipy.interpolate import CubicSpline
        interpolated_parameters = np.stack(
            [
                CubicSpline(source_positions, parameter_sequence, extrapolate=True, bc_type="natural")(target_positions)
                for parameter_sequence in parameters.T
            ], axis=1,
        )
        if len is not None:
            interpolated_parameters = interpolated_parameters[:max_length]

        result = AffineStack.from_array(
            interpolated_parameters.astype(self.dtype),
            pivot=self.pivot,
            sequenced=True,
            metadata=self.metadata,
        )
        result.set_metadata("z_step", 1)

        return result

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def smooth_gaussian(self, sigma, mode="reflect"):
        """
        Smooth affine parameters along the stack axis with a Gaussian filter.
        """
        sigma = float(sigma)

        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be a finite non-negative value.")

        if len(self) == 0 or sigma == 0:
            return self.copy()

        parameters = self.as_array(homogeneous=False, flatten=True).astype(np.float64)

        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(
            parameters,
            sigma=sigma, axis=0, mode=mode,
        )

        return AffineStack.from_array(
            smoothed.astype(self.dtype),
            pivot=self.pivot, sequenced=self.sequenced, metadata=self.metadata,
        )

    def smooth_median(
        self,
        radius,
        mode="nearest",
    ):
        """
        Smooth affine parameters along the stack axis with a median filter.
        The filter window size is ``2 * radius + 1``.
        """
        if not isinstance(radius, (int, np.integer)):
            raise TypeError("radius must be an integer.")
        if radius < 0:
            raise ValueError("radius must be non-negative.")

        if len(self) == 0 or radius == 0:
            return self.copy()

        parameters = self.as_array(homogeneous=False, flatten=True)

        from scipy.ndimage import median_filter
        smoothed = median_filter(parameters, size=(2 * radius + 1, 1), mode=mode)

        return AffineStack.from_array(
            smoothed,
            pivot=self.pivot,
            sequenced=self.sequenced,
            metadata=self.metadata,
        )

    @staticmethod
    def _z_interpolate(stack, scale):

        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be a finite value greater than zero.")

        stack = np.asarray(stack)
        if stack.ndim != 2:
            raise ValueError("stack must have shape (n_transforms, n_parameters).")

        if len(stack) == 0 or scale == 1:
            return stack.copy()

        dtype = stack.dtype
        working = stack.astype(np.float64)

        inverse_scale = 1.0 / scale
        integer_step = round(inverse_scale)

        if (scale < 1 and np.isclose(inverse_scale, integer_step)):
            result = working[::integer_step]
        else:
            from scipy.ndimage import zoom
            result = zoom(working, zoom=(scale, 1.0), order=1, grid_mode=False, mode="nearest")

        return result.astype(dtype, copy=False)

    def interpolate(self, scale):
        """
        Resample the affine stack along its stack axis.
        The affine parameters themselves are linearly interpolated.
        """
        if len(self) == 0:
            return self.copy()

        interpolated = self._z_interpolate(self.as_array(homogeneous=False, flatten=True), scale)

        return AffineStack.from_array(
            interpolated,
            pivot=self.pivot,
            sequenced=self.sequenced,
            metadata=self.metadata,
        )

    def scaled_for_stack_resize(self, scale):
        """
        Adapt a sequenced affine stack to uniformly resized stack data.

        The transforms are adjusted in the spatial dimensions and the stack is
        resampled along the z-axis using the same scale factor.
        """
        if not self.sequenced:
            raise ValueError("Scaling for stack resize requires a sequenced stack.")

        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be a finite value greater than zero.")

        if len(self) == 0:
            return self.copy()

        scaled_parameters = np.stack([
            matrix.scaled_for_image_resize(scale).as_array(homogeneous=False, flatten=True)
            for matrix in self
        ])
        scaled_parameters = self._z_interpolate(scaled_parameters, scale)

        return AffineStack.from_array(
            scaled_parameters,
            pivot=self.pivot * scale,
            sequenced=True,
            metadata=self.metadata,
        )

    def replace_large_translations(self, max_distance, replacement=None):
        """
        Replace transforms whose translation magnitude exceeds
        ``max_distance``.

        Parameters
        ----------
        max_distance
            Maximum allowed Euclidean translation magnitude.

        replacement
            Replacement transform. If None, the identity transform is used.

        Returns
        -------
        AffineStack
        """
        max_distance = float(max_distance)

        if not np.isfinite(max_distance) or max_distance < 0:
            raise ValueError("max_distance must be a finite non-negative value.")

        if replacement is None:
            replacement = AffineMatrix.identity(self.ndim, pivot=self.pivot, dtype=self.dtype)

        elif not isinstance(replacement, AffineMatrix):
            raise TypeError("replacement must be an AffineMatrix.")

        return AffineStack(
            matrices=[
                replacement.copy() if np.linalg.norm(matrix.translation) > max_distance else matrix.copy()
                for matrix in self
            ],
            sequenced=self.sequenced,
            metadata=self.metadata,
        )

    def auto_pad(self, stack_bounds=None, extra_padding=0):
        """
        Shift all transforms so that the transformed slice content fits into
        a common positive-coordinate canvas.

        Parameters
        ----------
        stack_bounds
            Per-slice bounds in the form ``[y, x, height, width]``.

            If None, the metadata entry ``"bounds"`` is used.

        extra_padding
            Additional padding added on all four canvas edges.

        Returns
        -------
        AffineStack
            New shifted affine stack.

        numpy.ndarray
            New spatial output shape ``[height, width]``.
        """
        if self.ndim != 2:
            raise ValueError("auto_pad currently only supports 2D affine transforms.")
        if len(self) == 0:
            raise ValueError("Cannot auto-pad an empty AffineStack.")

        if stack_bounds is None:
            if not self.has_metadata("bounds"):
                raise ValueError("stack_bounds was not supplied and metadata contains no 'bounds' entry.")

            stack_bounds = self.get_metadata("bounds")

        stack_bounds = np.asarray(stack_bounds, dtype=float)
        if stack_bounds.shape != (len(self), 4):
            raise ValueError(f"stack_bounds must have shape ({len(self)}, 4); received {stack_bounds.shape}.")
        if not np.isfinite(stack_bounds).all():
            raise ValueError("stack_bounds values must all be finite.")
        if np.any(stack_bounds[:, 2:] < 0):
            raise ValueError("Bounds height and width must be non-negative.")

        extra_padding = float(extra_padding)
        if not np.isfinite(extra_padding) or extra_padding < 0:
            raise ValueError("extra_padding must be a finite non-negative value.")

        def transform_bounds(matrix, bounds):
            y, x, height, width = bounds

            corners = np.array([[y, x], [y, x + width], [y + height, x], [y + height, x + width]], dtype=float)
            transformed_corners = matrix.inverse().apply(corners)

            return np.array([transformed_corners.min(axis=0), transformed_corners.max(axis=0)])

        transformed_bounds = np.array(
            [transform_bounds(matrix, stack_bounds[index]) for index, matrix in enumerate(self)]
        )

        global_minimum = transformed_bounds[:, 0].min(axis=0)
        global_maximum = transformed_bounds[:, 1].max(axis=0)

        shift = global_minimum - extra_padding
        shift_matrix = AffineMatrix.from_translation(shift, pivot=self.pivot, dtype=self.dtype)

        shifted_stack = AffineStack(
            matrices=[matrix @ shift_matrix for matrix in self],
            sequenced=self.sequenced,
            metadata=self.metadata,
        )

        output_shape = [
            len(self),
            *np.ceil(global_maximum - global_minimum + 2 * extra_padding).astype(int).tolist(),
        ]

        shifted_stack.set_metadata('stack_shape', output_shape)

        return shifted_stack, output_shape
    
    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self, name=None, default=None):
        if name is None:
            return self.metadata
        return deepcopy(self._metadata.get(name, default))

    def set_metadata(self, name=None, value=None, data=None):
        if data is not None:
            if name is not None or value is not None:
                raise ValueError("Specify either 'data' or 'name'/'value', not both.")

            self._metadata = self._normalize_metadata(data)
            return
        
        if name is None:
            raise ValueError("'name' must be specified unless 'data' is given.")

        self._metadata[name] = deepcopy(value)

    def has_metadata(self, name):
        return name in self._metadata

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def write(self, filepath):
        from pathlib import Path
        import json

        path = Path(filepath)

        data = {
            "transforms": self.as_array(homogeneous=False, flatten=True).tolist(),
            "sequenced": self.sequenced,
            "metadata": _jsonify(self.metadata),
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def read(cls, filepath):
        from pathlib import Path
        import json

        path = Path(filepath)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        transforms = data["transforms"]

        if len(transforms) == 0:
            return cls(sequenced=data.get("sequenced", False), metadata=data.get("metadata", {}))

        return cls.from_array(
            transforms,
            sequenced=data.get("sequenced", False),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def read_many(cls, filepaths, sequence=False):
        filepaths = list(filepaths)

        if len(filepaths) == 0:
            return cls()

        stacks = [
            cls.read(filepath)
            for filepath in filepaths
        ]

        if not sequence:
            result = stacks[0].copy()
            for stack in stacks[1:]:
                result.extend(stack)
            return result

        first = (stacks[0] if stacks[0].sequenced else stacks[0].to_sequenced())
        result = first.copy()

        for stack in stacks[1:]:
            current = (stack if stack.sequenced else stack.to_sequenced())
            offset = result[-1]
            result.extend(
                AffineStack(
                    matrices=[matrix @ offset for matrix in current],
                    sequenced=True,
                    metadata=current.metadata,
                )
            )

        return result
    
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @classmethod
    def _normalize_matrices(
        cls,
        matrices: Iterable[AffineMatrix | np.typing.ArrayLike],
        pivot: np.typing.ArrayLike | None = None,
    ) -> list[AffineMatrix]:
        """
        Normalize an iterable into a list of independent AffineMatrix objects.

        Raw matrix arrays are converted through AffineMatrix.from_array().
        Existing AffineMatrix objects are copied.
        """
        if isinstance(matrices, np.ndarray):
            matrices = list(matrices)

        try:
            matrix_items = list(matrices)
        except TypeError as error:
            raise TypeError("matrices must be an iterable of AffineMatrix objects or affine arrays.") from error

        normalized = []

        for index, matrix in enumerate(matrix_items):
            if isinstance(matrix, AffineMatrix):
                normalized_matrix = matrix.copy()

                if pivot is not None:
                    requested_pivot = AffineMatrix._normalize_pivot(pivot, ndim=normalized_matrix.ndim)
                    if not np.allclose(normalized_matrix.pivot, requested_pivot):
                        raise ValueError(
                            f"Matrix at index {index} has pivot {normalized_matrix.pivot.tolist()}, but the "
                            f"requested stack pivot is {requested_pivot.tolist()}."
                        )
            else:
                try:
                    normalized_matrix = AffineMatrix.from_array(matrix, pivot=pivot)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Invalid affine matrix at stack index {index}.") from error

            normalized.append(normalized_matrix)

        return normalized

    @staticmethod
    def _validate_matrices(matrices: Sequence[AffineMatrix]) -> None:
        """
        Validate dimensionality and pivot consistency.
        """
        if not matrices:
            return
        if not all(isinstance(matrix, AffineMatrix) for matrix in matrices):
            raise TypeError("All stack entries must be AffineMatrix objects.")

        reference = matrices[0]

        for index, matrix in enumerate(matrices[1:], start=1):
            if matrix.ndim != reference.ndim:
                raise ValueError(
                    "All matrices in an AffineStack must have the same "
                    f"dimensionality. Matrix 0 is {reference.ndim}D, but matrix {index} is {matrix.ndim}D."
                )
            if not np.allclose(matrix.pivot, reference.pivot):
                raise ValueError(
                    "All matrices in an AffineStack must have the same pivot. Matrix 0 has pivot "
                    f"{reference.pivot.tolist()}, but matrix {index} has pivot {matrix.pivot.tolist()}."
                )

    @staticmethod
    def _validate_sequenced(sequenced: bool) -> bool:
        if not isinstance(sequenced, (bool, np.bool_)):
            raise TypeError("sequenced must be a boolean.")

        return bool(sequenced)

    @staticmethod
    def _normalize_metadata(metadata: dict | None) -> dict:
        if metadata is None:
            return {}
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary or None.")

        return deepcopy(metadata)

    # ------------------------------------------------------------------
    # Python protocol methods
    # ------------------------------------------------------------------

    def copy(self):
        """
        Return a deep copy of the affine stack.
        """
        return AffineStack(matrices=self, sequenced=self.sequenced, metadata=self.metadata)

    def __len__(self):
        return len(self._matrices)

    def __iter__(self):
        return iter(self._matrices)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.get_substack(item)
        return self._matrices[item]

    def __matmul__(self, other):
        """
        Compose the stack with an AffineMatrix or another AffineStack.

        ``self @ other`` applies ``other`` first and ``self`` second.
        """
        return self.composed_with(other)
    
    def __eq__(self, other):
        if not isinstance(other, AffineStack):
            return NotImplemented
        return (
            self.sequenced == other.sequenced
            and self.metadata == other.metadata
            and len(self) == len(other)
            and all(left == right for left, right in zip(self, other))
        )
    def __array__(self, dtype=None):
        array = self.as_homogeneous()
        if dtype is not None:
            array = array.astype(dtype)
        return array

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"length={len(self)}, "
            f"ndim={self.ndim}, "
            f"sequenced={self.sequenced})"
        )


class AffineMatrix:
    """
    Represents a single 2D or 3D affine transformation.

    Internally, the transformation is stored as a homogeneous matrix:

    2D:
        [[a, b, tx],
         [c, d, ty],
         [0, 0,  1]]

    3D:
        [[a, b, c, tx],
         [d, e, f, ty],
         [g, h, i, tz],
         [0, 0, 0,  1]]
    """

    def __init__(self, matrix: np.typing.ArrayLike, pivot: np.typing.ArrayLike | None = None) -> None:
        normalized = self._normalize_array(matrix)
        self._matrix = normalized
        self._ndim = normalized.shape[0] - 1
        self._pivot = self._normalize_pivot(pivot, ndim=self._ndim)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def identity(
        cls,
        ndim: int,
        pivot: np.typing.ArrayLike | None = None,
        dtype: np.dtype | type = float,
    ) -> AffineMatrix:
        cls._validate_ndim(ndim)
        return cls(np.eye(ndim + 1, dtype=dtype), pivot=pivot)

    @classmethod
    def from_array(
        cls,
        array: np.typing.ArrayLike,
        pivot: np.typing.ArrayLike | None = None,
    ) -> AffineMatrix:
        return cls(array, pivot=pivot)

    @classmethod
    def from_translation(
        cls,
        translation: np.typing.ArrayLike,
        pivot: np.typing.ArrayLike | None = None,
        dtype: np.dtype | type = float,
    ) -> AffineMatrix:
        translation_array = np.asarray(translation, dtype=dtype)

        if translation_array.ndim != 1:
            raise ValueError("Translation must be a one-dimensional sequence.")

        ndim = translation_array.size
        cls._validate_ndim(ndim)

        matrix = np.eye(ndim + 1, dtype=dtype)
        matrix[:ndim, ndim] = translation_array

        return cls(matrix, pivot=pivot)

    @classmethod
    def from_rotation(
        cls,
        rotation: np.typing.ArrayLike | float,
        pivot: np.typing.ArrayLike | None = None,
        dtype: np.dtype | type = float,
    ) -> AffineMatrix:
        rotation_array = np.asarray(rotation, dtype=dtype)

        # 2D angle in radians
        if rotation_array.ndim == 0:
            angle = float(rotation_array)

            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)

            matrix = np.array(
                [
                    [cos_angle, -sin_angle, 0.0],
                    [sin_angle, cos_angle, 0.0],
                    [0.0, 0.0, 1.0],
                ], dtype=dtype,
            )

            return cls(matrix, pivot=pivot)

        # Explicit 2D rotation matrix
        if rotation_array.shape == (2, 2):
            matrix = np.eye(3, dtype=dtype)
            matrix[:2, :2] = rotation_array
            return cls(matrix, pivot=pivot)

        # Explicit 3D rotation matrix
        if rotation_array.shape == (3, 3):
            matrix = np.eye(4, dtype=dtype)
            matrix[:3, :3] = rotation_array
            return cls(matrix, pivot=pivot)

        # 3D Euler angles in radians
        if rotation_array.shape == (3,):
            rx, ry, rz = rotation_array

            cx, sx = np.cos(rx), np.sin(rx)
            cy, sy = np.cos(ry), np.sin(ry)
            cz, sz = np.cos(rz), np.sin(rz)

            rotation_x = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, cx, -sx],
                    [0.0, sx, cx],
                ],
                dtype=dtype,
            )

            rotation_y = np.array(
                [
                    [cy, 0.0, sy],
                    [0.0, 1.0, 0.0],
                    [-sy, 0.0, cy],
                ],
                dtype=dtype,
            )

            rotation_z = np.array(
                [
                    [cz, -sz, 0.0],
                    [sz, cz, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=dtype,
            )

            rotation_matrix = rotation_z @ rotation_y @ rotation_x

            matrix = np.eye(4, dtype=dtype)
            matrix[:3, :3] = rotation_matrix

            return cls(matrix, pivot=pivot)

        raise ValueError(
            "Rotation must be one of the following:\n"
            "- a scalar angle in radians for 2D;\n"
            "- a 2x2 rotation matrix;\n"
            "- a 3x3 rotation matrix;\n"
            "- three Euler angles in radians."
        )

    @classmethod
    def from_scale(
        cls,
        scale: np.typing.ArrayLike | float,
        ndim: int | None = None,
        pivot: np.typing.ArrayLike | None = None,
        dtype: np.dtype | type = float,
    ) -> AffineMatrix:
        scale_array = np.asarray(scale, dtype=dtype)

        if scale_array.ndim == 0:
            if ndim is None: 
                raise ValueError("ndim must be supplied for uniform scalar scaling.")

            cls._validate_ndim(ndim)
            scale_values = np.full(ndim, scale_array, dtype=dtype)

        elif scale_array.ndim == 1:
            inferred_ndim = scale_array.size
            cls._validate_ndim(inferred_ndim)

            if ndim is not None and ndim != inferred_ndim:
                raise ValueError(f"Scale has {inferred_ndim} elements, but ndim={ndim}.")

            ndim = inferred_ndim
            scale_values = scale_array

        else:
            raise ValueError("Scale must be a scalar or a one-dimensional sequence.")

        matrix = np.eye(ndim + 1, dtype=dtype)
        matrix[np.arange(ndim), np.arange(ndim)] = scale_values

        return cls(matrix, pivot=pivot)

    @classmethod
    def from_elastix(cls, parameters, pivot=None):
        """
        Construct an AffineMatrix from Elastix parameters.

        Accepted inputs are:
        - ``[transform_name, transform_parameters]``
        - ``SimpleITK.ParameterMap``
        """
        try:
            import SimpleITK as sitk
        except ImportError:
            sitk = None

        if sitk is not None and isinstance(parameters, sitk.SimpleITK.ParameterMap):
            parameter_map = parameters

            transform = parameter_map["Transform"][0]
            transform_parameters = parameter_map["TransformParameters"]

            if pivot is None and "CenterOfRotationPoint" in parameter_map:
                pivot = [float(value) for value in parameter_map["CenterOfRotationPoint"]][::-1]

            parameters = [transform, transform_parameters]

            if pivot is None and "CenterOfRotationPoint" in parameters:
                pivot = [float(value) for value in parameters["CenterOfRotationPoint"]][::-1]

        if not isinstance(parameters, (list, tuple)):
            raise TypeError("Elastix parameters must be a ParameterMap or [transform_name, transform_parameters].")
        if len(parameters) != 2:
            raise ValueError("Elastix parameters must contain exactly two entries: [transform_name, transform_parameters].")

        transform, transform_parameters = parameters
        if not isinstance(transform, str):
            raise TypeError("The Elastix transform name must be a string.")

        valid_transforms = {
            "translation",
            "TranslationTransform",
            "rigid",
            "EulerTransform",
            "SimilarityTransform",
            "affine",
            "AffineTransform",
        }

        if transform not in valid_transforms:
            raise ValueError(f"Unsupported Elastix transform: {transform!r}.")
        if transform == "TranslationTransform":
            transform = "translation"

        from squirrel.library.elastix import elastix_to_c
        compact = elastix_to_c(transform, transform_parameters)

        return cls.from_array(compact, pivot=pivot)
        
    @classmethod
    def read(cls, filepath) -> AffineMatrix:
        """
        Read an affine transformation from a JSON or CSV file.
        """
        from pathlib import Path

        path = Path(filepath)
        suffix = path.suffix.lower()

        if suffix == ".json":
            import json

            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if "transform" not in data:
                raise ValueError("Affine JSON data must contain a 'transform' field.")

            return cls.from_array(data["transform"], pivot=data.get("pivot"))

        if suffix == ".csv":
            matrix = np.genfromtxt(path, delimiter=",")
            return cls.from_array(matrix)

        raise ValueError(f"Unsupported file type. Expected a '.json' or '.csv' file; received '{path.suffix}'.")
    
    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def pivot(self) -> np.ndarray:
        return self._pivot.copy()

    @property
    def dtype(self):
        return self._matrix.dtype

    @property
    def linear(self) -> np.ndarray:
        """
        The linear (rotation/scale/shear) part of the affine transform.
        """
        return self._matrix[:self.ndim, :self.ndim].copy()

    @property
    def translation(self) -> np.ndarray:
        """
        The translation vector of the affine transform.
        """
        return self._matrix[:self.ndim, self.ndim].copy()

    @property
    def matrix(self):
        return self._matrix.copy()

    # ------------------------------------------------------------------
    # Array conversion
    # ------------------------------------------------------------------

    def as_array(
        self,
        homogeneous: bool = True,
        flatten: bool = False,
        copy: bool = True,
    ):
        """
        Return the affine transformation as a NumPy array.

        Parameters
        ----------
        homogeneous
            If True, return the full homogeneous matrix.
            Otherwise return the compact affine matrix.

        flatten
            If True, flatten the returned array.

        copy
            If True (default), return a copy. Otherwise, return a view whenever
            possible.
        """
        if homogeneous:
            matrix = self._matrix
        else:
            matrix = self._matrix[:-1]

        if copy:
            matrix = matrix.copy()

        if flatten:
            matrix = matrix.reshape(-1)

        return matrix

    def as_compact(self):
        """
        Return the affine matrix without the final homogeneous row.

        2D: shape (2, 3)
        3D: shape (3, 4)
        """
        return self.as_array(homogeneous=False)

    def as_homogeneous(self):
        """
        Return the full homogeneous matrix.

        2D: shape (3, 3)
        3D: shape (4, 4)
        """
        return self.as_array(homogeneous=True)

    def as_scipy_affine(self):
        """
        Return the matrix and offset expected by
        scipy.ndimage.affine_transform().

        The AffineMatrix describes the forward transformation, while SciPy
        expects the inverse mapping from output coordinates to input coordinates.
        """
        inverse = self.shifted_pivot_to_origin().inverse()

        return (inverse.linear, inverse.translation)

    def to_elastix(self, shape=None, as_parameter_map=False):
        """
        Convert the affine transformation to Elastix parameters.

        Parameters
        ----------
        shape
            Optional image shape. Required only when constructing a complete
            SimpleITK ParameterMap with image-size information.

        as_parameter_map
            If False, return the Elastix affine parameter array.
            If True, return a SimpleITK ParameterMap.

        Returns
        -------
        numpy.ndarray or SimpleITK.ParameterMap
            Elastix affine parameters or a complete parameter map.
        """
        from squirrel.library.elastix import c_to_elastix

        elastix_parameters = c_to_elastix(self.as_compact().astype(np.float64).reshape(-1))

        if not as_parameter_map:
            return elastix_parameters

        try:
            import SimpleITK as sitk
        except ImportError as error:
            raise ImportError("Creating an Elastix ParameterMap requires SimpleITK.") from error

        parameter_map = sitk.ParameterMap()
        parameter_map["Transform"] = ["AffineTransform"]
        parameter_map["TransformParameters"] = [str(value) for value in elastix_parameters]
        parameter_map["NumberOfParameters"] = [str(len(elastix_parameters))]
        # Elastix uses reversed spatial axis order relative to the
        # internal NumPy representation used here.
        parameter_map["CenterOfRotationPoint"] = [str(value) for value in self.pivot[::-1]]
        parameter_map["Spacing"] = ["1"] * self.ndim
        parameter_map["Index"] = ["0"] * self.ndim
        parameter_map["Origin"] = ["0"] * self.ndim
        parameter_map["Direction"] = [str(value) for value in np.eye(self.ndim).reshape(-1)]
        parameter_map["UseDirectionCosines"] = ["true"]

        if shape is not None:
            shape_array = np.asarray(shape)

            if shape_array.shape != (self.ndim,):
                raise ValueError(f"shape must have length {self.ndim}; received shape {shape_array.shape}.")
            if not np.issubdtype(shape_array.dtype, np.integer):
                if not np.all(np.equal(shape_array, np.floor(shape_array))):
                    raise ValueError("shape values must be integers.")
            if np.any(shape_array <= 0):
                raise ValueError("shape values must be positive.")

            parameter_map["Size"] = [str(int(value)) for value in shape_array[::-1]]

        return parameter_map

    def write(self, filepath) -> None:
        """
        Write the affine transformation to a JSON or CSV file.

        JSON stores both the compact transformation and its pivot.
        CSV stores only the compact transformation because CSV has no
        natural place for additional pivot metadata.

        Parameters
        ----------
        filepath
            Output path ending in ``.json`` or ``.csv``.
        """
        from pathlib import Path

        path = Path(filepath)
        suffix = path.suffix.lower()

        if suffix == ".json":
            import json
            data = {
                "transform": (self.as_compact().astype(np.float64).reshape(-1).tolist()),
                "pivot": (self.pivot.astype(np.float64).tolist()),
            }
            with path.open("w", encoding="utf-8") as file:
                json.dump(data, file, indent=2)
            return

        if suffix == ".csv":
            np.savetxt(path, self.as_compact().astype(np.float64), delimiter=",")
            return

        raise ValueError(f"Unsupported file type. Expected a '.json' or '.csv' file; received '{path.suffix}'.")

    # ------------------------------------------------------------------
    # Transformation operations
    # ------------------------------------------------------------------

    def copy(self) -> AffineMatrix:
        return AffineMatrix(matrix=self._matrix, pivot=self.pivot)

    def compose(self, other: AffineMatrix) -> AffineMatrix:
        """
        Compose this transformation with another transformation.

        ``self.compose(other)`` applies ``other`` first and ``self`` second.

        Both transformations must have the same dimensionality and pivot.
        """
        if not isinstance(other, AffineMatrix):
            raise TypeError(f"Can only compose with another AffineMatrix; received {type(other).__name__}.")
        if self.ndim != other.ndim:
            raise ValueError(f"Cannot compose affine transformations with different dimensions: {self.ndim} and {other.ndim}.")
        if not np.allclose(self.pivot, other.pivot):
            raise ValueError(
                f"Cannot compose affine transformations with different pivots: {self.pivot.tolist()} and {other.pivot.tolist()}."
            )

        matrix = self._matrix @ other._matrix
        return AffineMatrix(matrix=matrix, pivot=self.pivot)

    def inverse(self) -> AffineMatrix:
        """
        Return the inverse affine transformation.
        """
        try:
            matrix = np.linalg.inv(self._matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError("The affine transformation is singular and cannot be inverted.") from error
        return AffineMatrix(matrix=matrix, pivot=self.pivot)

    def apply(self, points: np.typing.ArrayLike) -> np.typing.NDArray[np.floating]:
        """
        Apply the affine transformation to one or more points.

        Accepted point shapes are:

        - ``(ndim,)`` for one point
        - ``(..., ndim)`` for multiple points

        The pivot is included in the transformation:

            transformed = A @ (point - pivot) + pivot + translation

        Parameters
        ----------
        points:
            A single point or an array of points.

        Returns
        -------
        numpy.ndarray
            The transformed points with the same shape as the input.
        """
        points_array = np.asarray(points)

        if not np.issubdtype(points_array.dtype, np.number):
            raise TypeError("Point coordinates must be numeric.")
        if points_array.ndim == 0:
            raise ValueError("Points must have shape (ndim,) or (..., ndim).")
        if points_array.shape[-1] != self.ndim:
            raise ValueError(f"The last point dimension must be {self.ndim}; received shape {points_array.shape}.")

        dtype = np.result_type(points_array.dtype, self.dtype)
        points_array = points_array.astype(dtype, copy=False)

        if not np.isfinite(points_array).all():
            raise ValueError("Point coordinates must all be finite.")

        return ((points_array - self.pivot) @ self.linear.T + self.pivot + self.translation)

    def with_translation(self, translation: np.typing.ArrayLike) -> AffineMatrix:
        """
        Return a copy with a replaced translation vector.

        The original AffineMatrix is not modified.
        """
        translation_array = np.asarray(translation, dtype=self.dtype)

        if translation_array.shape != (self.ndim,):
            raise ValueError(f"Translation must have shape ({self.ndim},); received shape {translation_array.shape}.")
        if not np.isfinite(translation_array).all():
            raise ValueError("Translation values must all be finite.")

        matrix = self._matrix.copy()
        matrix[:self.ndim, self.ndim] = translation_array

        return AffineMatrix(matrix=matrix, pivot=self.pivot)

    def with_pivot(
        self,
        pivot: np.typing.ArrayLike,
    ) -> AffineMatrix:
        """
        Return a copy with a replaced pivot.

        Note that replacing the pivot changes the effective transformation.
        The stored matrix itself is not adjusted.
        """
        return AffineMatrix(matrix=self._matrix, pivot=pivot)

    def shifted_pivot_to_origin(self) -> AffineMatrix:
        """
        Return an equivalent transformation whose pivot is the origin.

        The translation is adjusted so that applying the returned
        transformation gives the same result as applying the original one.

        For a transformation
            y = A @ (x - p) + p + t

        the equivalent origin-based translation is
            t_origin = t + p - A @ p
        """
        pivot = self.pivot

        translation_at_origin = (self.translation + pivot - self.linear @ pivot)

        matrix = self._matrix.copy()
        matrix[:self.ndim, self.ndim] = translation_at_origin

        return AffineMatrix(matrix=matrix, pivot=np.zeros(self.ndim, dtype=self.dtype))

    def scaled_for_image_resize(self, scale: np.typing.ArrayLike | float) -> AffineMatrix:
        """
        Adapt the transformation to resized image coordinates.

        A scalar applies uniform scaling. A vector applies one scale factor
        per spatial dimension.

        For anisotropic scaling represented by a diagonal matrix D:

            A_scaled = D @ A @ inv(D)
            t_scaled = D @ t
            p_scaled = D @ p

        For uniform scaling, the linear component remains unchanged.
        """
        scale_array = np.asarray(scale, dtype=self.dtype)

        if scale_array.ndim == 0:
            scale_values = np.full(self.ndim, scale_array, dtype=self.dtype)
        elif scale_array.shape == (self.ndim,):
            scale_values = scale_array
        else:
            raise ValueError(f"Scale must be a scalar or have shape ({self.ndim},); received shape {scale_array.shape}.")
        if not np.isfinite(scale_values).all():
            raise ValueError("Scale values must all be finite.")
        if np.any(scale_values == 0):
            raise ValueError("Scale values must be non-zero.")

        coordinate_scale = np.diag(scale_values)
        inverse_coordinate_scale = np.diag(1.0 / scale_values)

        scaled_linear = (coordinate_scale @ self.linear @ inverse_coordinate_scale)
        scaled_translation = coordinate_scale @ self.translation
        scaled_pivot = scale_values * self.pivot

        matrix = np.eye(self.ndim + 1, dtype=np.result_type(self.dtype, scale_values.dtype))
        matrix[:self.ndim, :self.ndim] = scaled_linear
        matrix[:self.ndim, self.ndim] = scaled_translation

        return AffineMatrix(matrix=matrix, pivot=scaled_pivot)

    def to_3d(self, axis: int = 2) -> AffineMatrix:
        """
        Embed a 2D affine transformation into three dimensions.

        Parameters
        ----------
        axis:
            The axis orthogonal to the embedded 2D plane.

            - ``axis=0`` embeds the transform in the YZ plane
            - ``axis=1`` embeds the transform in the XZ plane
            - ``axis=2`` embeds the transform in the XY plane
        """
        if self.ndim == 3:
            return self.copy()

        if self.ndim != 2:
            raise ValueError("Only a 2D affine transformation can be embedded in 3D.")

        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2; received {axis}.")

        active_axes = [idx for idx in range(3) if idx != axis]

        matrix = np.eye(4, dtype=self.dtype)
        matrix[np.ix_(active_axes, active_axes)] = self.linear
        matrix[active_axes, 3] = self.translation

        pivot = np.zeros(3, dtype=self.pivot.dtype)
        pivot[active_axes] = self.pivot

        return AffineMatrix(matrix=matrix, pivot=pivot)

    def decompose(
        self,
    ) -> tuple[AffineMatrix, AffineMatrix, AffineMatrix, AffineMatrix]:
        try:
            from transforms3d.affines import decompose
        except ImportError as error:
            raise ImportError("Affine decomposition requires the 'transforms3d' package.") from error

        if self.ndim == 2:
            affine = self.to_3d(axis=2).shifted_pivot_to_origin()
        else:
            affine = self.shifted_pivot_to_origin()

        translation, rotation, scale, shear = decompose(affine._matrix.astype(np.float64))

        if self.ndim == 2:
            translation = translation[:2]
            rotation = rotation[:2, :2]
            scale = scale[:2]
            shear = shear[:1]

        ndim = self.ndim
        zero_pivot = np.zeros(ndim, dtype=self.dtype)

        translation_matrix = np.eye(ndim + 1, dtype=self.dtype)
        translation_matrix[:ndim, ndim] = translation

        rotation_matrix = np.eye(ndim + 1, dtype=self.dtype)
        rotation_matrix[:ndim, :ndim] = rotation

        scale_matrix = np.eye(ndim + 1, dtype=self.dtype)
        scale_matrix[np.arange(ndim), np.arange(ndim)] = scale

        shear_matrix = np.eye(ndim + 1, dtype=self.dtype)

        if ndim == 2:
            shear_matrix[0, 1] = shear[0]
        else:
            shear_matrix[0, 1] = shear[0]
            shear_matrix[0, 2] = shear[1]
            shear_matrix[1, 2] = shear[2]

        return (
            AffineMatrix(translation_matrix, pivot=zero_pivot),
            AffineMatrix(rotation_matrix, pivot=zero_pivot),
            AffineMatrix(scale_matrix, pivot=zero_pivot),
            AffineMatrix(shear_matrix, pivot=zero_pivot),
        )

    # ------------------------------------------------------------------
    # Validation and normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_ndim(ndim: int) -> None:
        if ndim not in (2, 3):
            raise ValueError(f"Only 2D and 3D affine transformations are supported; received ndim={ndim}.")

    @classmethod
    def _normalize_array(cls, array: np.typing.ArrayLike) -> np.typing.NDArray[np.floating]:

        matrix = np.asarray(array)

        if not np.issubdtype(matrix.dtype, np.number):
            raise TypeError("Affine matrix values must be numeric.")

        if np.issubdtype(matrix.dtype, np.floating):
            matrix = matrix.copy()
        else:
            matrix = matrix.astype(np.float64)

        matrix = np.squeeze(matrix)

        # Flat compact 2D
        if matrix.ndim == 1 and matrix.size == 6:
            matrix = matrix.reshape(2, 3)

        # Flat homogeneous 2D
        elif matrix.ndim == 1 and matrix.size == 9:
            matrix = matrix.reshape(3, 3)

        # Flat compact 3D
        elif matrix.ndim == 1 and matrix.size == 12:
            matrix = matrix.reshape(3, 4)

        # Flat homogeneous 3D
        elif matrix.ndim == 1 and matrix.size == 16:
            matrix = matrix.reshape(4, 4)

        if matrix.shape == (2, 3):
            homogeneous = np.eye(3, dtype=matrix.dtype)
            homogeneous[:2, :] = matrix
            matrix = homogeneous

        elif matrix.shape == (3, 4):
            homogeneous = np.eye(4, dtype=matrix.dtype)
            homogeneous[:3, :] = matrix
            matrix = homogeneous

        elif matrix.shape not in ((3, 3), (4, 4)):
            raise ValueError(
                "Invalid affine matrix shape. Expected one of:\n"
                "- 6 values or shape (2, 3);\n"
                "- 9 values or shape (3, 3);\n"
                "- 12 values or shape (3, 4);\n"
                "- 16 values or shape (4, 4).\n"
                f"Received shape {matrix.shape}."
            )

        cls._validate_homogeneous_matrix(matrix)

        return matrix

    @staticmethod
    def _validate_homogeneous_matrix(
        matrix: np.NDArray[np.floating],
    ) -> None:
        ndim = matrix.shape[0] - 1

        expected_last_row = np.zeros(ndim + 1, dtype=matrix.dtype)
        expected_last_row[-1] = 1.0

        if not np.allclose(matrix[-1], expected_last_row):
            raise ValueError(
                "The final row of a homogeneous affine matrix must be "
                f"{expected_last_row.tolist()}; received {matrix[-1].tolist()}."
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Affine matrix values must all be finite.")

    @staticmethod
    def _normalize_pivot(
        pivot: np.typing.ArrayLike | None,
        ndim: int,
    ) -> np.NDArray[np.floating]:
        if pivot is None:
            return np.zeros(ndim, dtype=float)

        pivot_array = np.asarray(pivot, dtype=float)

        if pivot_array.shape != (ndim,):
            raise ValueError(f"Pivot must have shape ({ndim},); received shape {pivot_array.shape}.")
        if not np.isfinite(pivot_array).all():
            raise ValueError("Pivot values must all be finite.")

        return pivot_array.copy()

    # ------------------------------------------------------------------
    # Python protocol methods
    # ------------------------------------------------------------------

    def copy(self) -> AffineMatrix:
        """
        Return a deep copy of the affine transformation.
        """
        return AffineMatrix(matrix=self._matrix, pivot=self.pivot)

    def __matmul__(self, other: AffineMatrix) -> AffineMatrix:
        """
        Compose two affine transformations.
        """
        return self.compose(other)

    def __eq__(self, other) -> bool:
        """
        Compare two affine transformations for numerical equality.
        """
        if not isinstance(other, AffineMatrix):
            return NotImplemented

        return (
            self.ndim == other.ndim
            and np.allclose(self._matrix, other._matrix)
            and np.allclose(self.pivot, other.pivot)
        )

    def __array__(self, dtype=None):
        """
        Return the homogeneous matrix representation.

        Enables, for example,

            np.asarray(affine)
            np.linalg.inv(affine)
        """
        matrix = self._matrix.copy()

        if dtype is not None:
            matrix = matrix.astype(dtype)

        return matrix

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(ndim={self.ndim}, pivot={self.pivot.tolist()})")


if __name__ == '__main__':
    pass

