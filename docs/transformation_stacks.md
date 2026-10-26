# Transformation stack APIs

`AffineStack` and `ElastixStack` are ordered collections of transformations.
Both support `len(stack)`, iteration, integer indexing, slicing, `append`,
`extend`, `copy`, `read`, `read_many`, and `write`.

## AffineStack

```python
from squirrel.library.affine_matrices import AffineStack

stack = AffineStack(matrices=matrices, sequenced=False)
first = stack[0]        # AffineMatrix
subset = stack[2:5]     # AffineStack
stack.append(matrix)
stack.extend(other_stack)
copied = stack.copy()
stack.write('affines.json')
restored = AffineStack.read('affines.json')
```

AffineStack supports stack-level metadata and a `sequenced` flag that
distinguishes relative transforms from transforms in a common reference
frame. It also provides affine-specific composition and geometry methods.

## ElastixStack

```python
from squirrel.library.elastix import ElastixStack

stack = ElastixStack(transforms=parameter_maps)
first = stack[0]        # SimpleITK ParameterMap
subset = stack[2:5]     # ElastixStack
stack.append(parameter_map)
stack.extend(other_stack)
copied = stack.copy()
stack.write('elastix_transforms')
restored = ElastixStack.read('elastix_transforms')
combined = ElastixStack.read_many(['registration_a', 'registration_b'])
```

`ElastixStack.read` reads files matching `*.txt` from a directory in
lexicographic filename order. Pass `pattern=` to choose another glob.
`write` writes `transform_00000.txt`, `transform_00001.txt`, and so on.
The `image_shape` property reads the common `Size` parameter in NumPy
axis order; it returns `None` for an empty stack.

Slices and copies of either stack are independent collections. For
ElastixStack, individual parameter maps returned by integer indexing
remain mutable objects belonging to the stack; use `copy()` before
modifying them if the original must be preserved.

## Differences

AffineStack stores affine matrices and supports affine-specific
operations such as inversion, composition, and sequence conversion.
ElastixStack stores Elastix parameter maps, which can represent
non-affine transformations. The two classes intentionally do not
expose identical transformation mathematics.

The previous `get_substack`, ElastixStack `stack=` and `dirpath=`
constructor arguments, `set_from_stack`, `set_from_dir`, `to_file`,
and `load_transform_stack_from_multiple_files` interfaces are removed.
Use slicing, `transforms=`, `read`, `write`, and `read_many` instead.
