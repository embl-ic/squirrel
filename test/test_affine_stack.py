
import unittest
import warnings

import numpy as np

from squirrel.library.affine_matrices import (
    AffineMatrix,
    AffineStack,
)


class TestAffineStack(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_construct_from_affine_matrices(self):
        print(f'Testing AffineStack: construct from AffineMatrix objects ...')
        matrices = [
            AffineMatrix.identity(2),
            AffineMatrix.from_translation([10.0, 20.0]),
        ]

        stack = AffineStack(matrices)

        self.assertEqual(len(stack), 2)
        self.assertEqual(stack.ndim, 2)

    def test_constructor_copies_matrices(self):
        print(f'Testing AffineStack: constructor copies matrices ...')
        matrix = AffineMatrix.identity(2)
        stack = AffineStack([matrix])

        matrix._matrix[0, 0] = 10.0

        self.assertEqual(
            stack[0].linear[0, 0],
            1.0,
        )

    def test_empty_constructor(self):
        print('Testing AffineStack: empty constructor ...')

        stack = AffineStack()

        self.assertEqual(len(stack), 0)
        self.assertIsNone(stack.ndim)
        self.assertIsNone(stack.pivot)
        self.assertIsNone(stack.dtype)
        self.assertFalse(stack.sequenced)
        self.assertEqual(stack.metadata, {})

    def test_identity_2d(self):
        print(f'Testing AffineStack: identity 2D stack ...')
        stack = AffineStack.identity(
            length=3,
            ndim=2,
        )

        self.assertEqual(len(stack), 3)
        self.assertEqual(stack.ndim, 2)

        for matrix in stack:
            np.testing.assert_allclose(
                matrix.as_homogeneous(),
                np.eye(3),
            )

    def test_identity_3d(self):
        print(f'Testing AffineStack: identity 3D stack ...')
        stack = AffineStack.identity(
            length=2,
            ndim=3,
            pivot=[1.0, 2.0, 3.0],
        )

        self.assertEqual(len(stack), 2)
        self.assertEqual(stack.ndim, 3)

        for matrix in stack:
            np.testing.assert_allclose(
                matrix.pivot,
                [1.0, 2.0, 3.0],
            )

    def test_identity_zero_length(self):
        print('Testing AffineStack: identity zero-length stack ...')

        stack = AffineStack.identity(
            length=0,
            ndim=2,
        )

        self.assertEqual(len(stack), 0)
        self.assertIsNone(stack.ndim)
        self.assertIsNone(stack.pivot)
        self.assertIsNone(stack.dtype)

    def test_identity_rejects_negative_length(self):
        print(f'Testing AffineStack: identity rejects negative length ...')
        with self.assertRaisesRegex(
            ValueError,
            "non-negative",
        ):
            AffineStack.identity(
                length=-1,
                ndim=2,
            )

    def test_from_array_compact_2d(self):
        print(f'Testing AffineStack: from array compact 2D ...')
        array = np.array(
            [
                [
                    [1.0, 0.0, 10.0],
                    [0.0, 1.0, 20.0],
                ],
                [
                    [1.0, 0.0, 30.0],
                    [0.0, 1.0, 40.0],
                ],
            ]
        )

        stack = AffineStack.from_array(array)

        self.assertEqual(len(stack), 2)
        self.assertEqual(stack.ndim, 2)

        np.testing.assert_allclose(
            stack[0].translation,
            [10.0, 20.0],
        )
        np.testing.assert_allclose(
            stack[1].translation,
            [30.0, 40.0],
        )

    def test_from_array_flat_compact_2d(self):
        print(f'Testing AffineStack: from array flat compact 2D ...')
        array = np.array(
            [
                [1.0, 0.0, 10.0, 0.0, 1.0, 20.0],
                [1.0, 0.0, 30.0, 0.0, 1.0, 40.0],
            ]
        )

        stack = AffineStack.from_array(array)

        self.assertEqual(len(stack), 2)

        np.testing.assert_allclose(
            stack[1].translation,
            [30.0, 40.0],
        )

    def test_from_array_homogeneous_3d(self):
        print(f'Testing AffineStack: from array homogeneous 3D ...')
        first = np.eye(4)
        second = np.eye(4)

        first[:3, 3] = [1.0, 2.0, 3.0]
        second[:3, 3] = [4.0, 5.0, 6.0]

        stack = AffineStack.from_array(
            np.stack([first, second])
        )

        self.assertEqual(stack.ndim, 3)

        np.testing.assert_allclose(
            stack[0].translation,
            [1.0, 2.0, 3.0],
        )

    def test_constructor_rejects_mixed_dimensions(self):
        print(f'Testing AffineStack: constructor rejects mixed dimensions ...')
        matrices = [
            AffineMatrix.identity(2),
            AffineMatrix.identity(3),
        ]

        with self.assertRaisesRegex(
            ValueError,
            "Expected a .*D affine matrix",
        ):
            AffineStack(matrices)

    def test_constructor_rejects_different_pivots(self):
        print(f'Testing AffineStack: constructor rejects different pivots ...')
        matrices = [
            AffineMatrix.identity(
                2,
                pivot=[0.0, 0.0],
            ),
            AffineMatrix.identity(
                2,
                pivot=[1.0, 0.0],
            ),
        ]

        with self.assertRaisesRegex(
            ValueError,
            "pivot does not match stack pivot",
        ):
            AffineStack(matrices)

    def test_constructor_rejects_non_boolean_sequenced(self):
        print(f'Testing AffineStack: constructor rejects non-boolean sequenced ...')
        with self.assertRaisesRegex(
            TypeError,
            "sequenced must be a boolean",
        ):
            AffineStack.identity(
                length=1,
                ndim=2,
                sequenced="yes",
            )

    def test_properties(self):
        print(f'Testing AffineStack: properties ...')
        stack = AffineStack.identity(
            length=2,
            ndim=2,
            pivot=[10.0, 20.0],
            sequenced=True,
            metadata={"source": "registration"},
            dtype=np.float32,
        )

        self.assertEqual(stack.ndim, 2)
        self.assertEqual(stack.dtype, np.float32)
        self.assertTrue(stack.sequenced)

        np.testing.assert_allclose(
            stack.pivot,
            [10.0, 20.0],
        )

        self.assertEqual(
            stack.metadata,
            {"source": "registration"},
        )

    def test_pivot_property_returns_copy(self):
        print(f'Testing AffineStack: pivot property returns copy ...')
        stack = AffineStack.identity(
            length=1,
            ndim=2,
            pivot=[1.0, 2.0],
        )

        pivot = stack.pivot
        pivot[0] = 100.0

        np.testing.assert_allclose(
            stack.pivot,
            [1.0, 2.0],
        )

    def test_metadata_property_returns_copy(self):
        print(f'Testing AffineStack: metadata property returns copy ...')
        stack = AffineStack.identity(
            length=1,
            ndim=2,
            metadata={
                "nested": {
                    "values": [1, 2, 3],
                }
            },
        )

        metadata = stack.metadata
        metadata["nested"]["values"][0] = 100

        self.assertEqual(
            stack.metadata["nested"]["values"],
            [1, 2, 3],
        )

    def test_append_initializes_stack(self):
        print('Testing AffineStack: append initializes stack ...')

        stack = AffineStack()

        stack.append(
            AffineMatrix.identity(
                ndim=2,
                pivot=[1., 2.],
                dtype=np.float32,
            )
        )

        self.assertEqual(len(stack), 1)
        self.assertEqual(stack.ndim, 2)
        self.assertEqual(stack.dtype, np.float32)

        np.testing.assert_allclose(
            stack.pivot,
            [1., 2.]
        )

    def test_append_rejects_wrong_dimension(self):
        print('Testing AffineStack: append rejects wrong dimension ...')

        stack = AffineStack.identity(
            length=1,
            ndim=2,
        )

        with self.assertRaises(ValueError):
            stack.append(
                AffineMatrix.identity(3)
            )

    def test_append_rejects_wrong_pivot(self):
        print('Testing AffineStack: append rejects wrong pivot ...')

        stack = AffineStack.identity(
            length=1,
            ndim=2,
            pivot=[0., 0.]
        )

        with self.assertRaises(ValueError):
            stack.append(
                AffineMatrix.identity(
                    2,
                    pivot=[1., 0.]
                )
            )

    def test_len(self):
        print('Testing AffineStack: len ...')

        stack = AffineStack.identity(
            length=3,
            ndim=2,
        )

        self.assertEqual(len(stack), 3)

    def test_getitem(self):
        print('Testing AffineStack: getitem ...')

        stack = AffineStack.from_array([
            [[1., 0., 10.],
            [0., 1., 20.]],
            [[1., 0., 30.],
            [0., 1., 40.]],
        ])

        np.testing.assert_allclose(
            stack[1].translation,
            [30., 40.]
        )

    def test_getitem_slice(self):
        print('Testing AffineStack: getitem slice ...')

        stack = AffineStack.identity(
            length=5,
            ndim=2,
        )

        substack = stack[1:4]

        self.assertIsInstance(
            substack,
            AffineStack,
        )

        self.assertEqual(len(substack), 3)
        self.assertEqual(substack.ndim, 2)

    def test_iter(self):
        print('Testing AffineStack: iter ...')

        stack = AffineStack.identity(
            length=3,
            ndim=2,
        )

        count = 0

        for matrix in stack:
            self.assertIsInstance(
                matrix,
                AffineMatrix,
            )
            count += 1

        self.assertEqual(count, 3)

    def test_copy(self):
        print('Testing AffineStack: copy ...')

        stack = AffineStack.from_array(
            [
                [[1., 0., 10.],
                [0., 1., 20.]],
                [[1., 0., 30.],
                [0., 1., 40.]],
            ],
            sequenced=True,
            metadata={"foo": "bar"},
        )

        copied = stack.copy()

        self.assertIsNot(copied, stack)
        self.assertEqual(len(copied), len(stack))
        self.assertEqual(copied.sequenced, stack.sequenced)
        self.assertEqual(copied.metadata, stack.metadata)

        copied[0]._matrix[0, 2] = 999.

        self.assertEqual(
            stack[0].translation[0],
            10.
        )

    def test_repr(self):
        print('Testing AffineStack: repr ...')

        stack = AffineStack.identity(
            length=3,
            ndim=2,
            sequenced=True,
        )

        self.assertEqual(
            repr(stack),
            "AffineStack(length=3, ndim=2, sequenced=True)"
        )

    def test_as_homogeneous(self):
        print('Testing AffineStack: as_homogeneous ...')

        stack = AffineStack.identity(
            length=2,
            ndim=2,
        )

        result = stack.as_homogeneous()

        self.assertEqual(result.shape, (2, 3, 3))

        np.testing.assert_allclose(
            result[0],
            np.eye(3),
        )

    def test_as_compact(self):
        print('Testing AffineStack: as_compact ...')

        stack = AffineStack.from_array([
            [[1., 0., 10.],
            [0., 1., 20.]],
            [[1., 0., 30.],
            [0., 1., 40.]],
        ])

        result = stack.as_compact()

        self.assertEqual(result.shape, (2, 2, 3))

        np.testing.assert_allclose(
            result[1],
            [[1., 0., 30.],
            [0., 1., 40.]]
        )

    def test_as_array_flatten(self):
        print('Testing AffineStack: as_array flatten ...')

        stack = AffineStack.identity(
            length=2,
            ndim=2,
        )

        result = stack.as_array(flatten=True)

        self.assertEqual(result.shape, (2, 9))

    def test_as_array_returns_copy(self):
        print('Testing AffineStack: as_array returns copy ...')

        stack = AffineStack.identity(
            length=1,
            ndim=2,
        )

        result = stack.as_array()
        result[0, 0, 0] = 999.

        self.assertEqual(
            stack[0].linear[0, 0],
            1.
        )

    def test_to_sequenced_identity(self):
        print('Testing AffineStack: to_sequenced identity ...')

        stack = AffineStack.identity(
            length=3,
            ndim=2,
        )

        sequenced = stack.to_sequenced()

        self.assertTrue(sequenced.sequenced)

        for matrix in sequenced:
            np.testing.assert_allclose(
                matrix.as_homogeneous(),
                np.eye(3),
            )

    def test_to_sequenced_translation(self):
        print('Testing AffineStack: to_sequenced translation ...')

        stack = AffineStack()

        stack.append(AffineMatrix.from_translation([1., 0.]))
        stack.append(AffineMatrix.from_translation([2., 0.]))
        stack.append(AffineMatrix.from_translation([3., 0.]))

        sequenced = stack.to_sequenced()

        np.testing.assert_allclose(
            sequenced[0].translation,
            [1., 0.]
        )
        np.testing.assert_allclose(
            sequenced[1].translation,
            [3., 0.]
        )
        np.testing.assert_allclose(
            sequenced[2].translation,
            [6., 0.]
        )

    def test_to_relative_translation(self):
        print('Testing AffineStack: to_relative translation ...')

        stack = AffineStack()

        stack.append(AffineMatrix.from_translation([1., 0.]))
        stack.append(AffineMatrix.from_translation([3., 0.]))
        stack.append(AffineMatrix.from_translation([6., 0.]))

        stack._sequenced = True

        relative = stack.to_relative()

        self.assertFalse(relative.sequenced)

        np.testing.assert_allclose(
            relative[0].translation,
            [1., 0.]
        )
        np.testing.assert_allclose(
            relative[1].translation,
            [2., 0.]
        )
        np.testing.assert_allclose(
            relative[2].translation,
            [3., 0.]
        )

    def test_to_relative_and_back(self):
        print('Testing AffineStack: to_relative and back ...')

        stack = AffineStack()

        stack.append(AffineMatrix.from_translation([1., 0.]))
        stack.append(AffineMatrix.from_translation([2., 0.]))
        stack.append(AffineMatrix.from_translation([3., 0.]))

        reconstructed = (
            stack
            .to_sequenced()
            .to_relative()
        )

        self.assertFalse(reconstructed.sequenced)

        self.assertEqual(len(stack), len(reconstructed))

        for original, new in zip(stack, reconstructed):
            self.assertEqual(original, new)

    def test_write_read(self):
        print('Testing AffineStack: write/read ...')

        import tempfile
        from pathlib import Path

        stack = AffineStack()

        stack.append(AffineMatrix.from_translation([1., 2.]))
        stack.append(AffineMatrix.from_translation([3., 4.]))

        stack._sequenced = True
        stack._metadata = {"foo": "bar"}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "stack.json"

            stack.write(filepath)
            loaded = AffineStack.read(filepath)

        self.assertEqual(len(stack), len(loaded))
        self.assertEqual(stack.sequenced, loaded.sequenced)
        self.assertEqual(stack.metadata, loaded.metadata)

        for a, b in zip(stack, loaded):
            self.assertEqual(a, b)

    def test_write_read_empty(self):
        print('Testing AffineStack: write/read empty stack ...')

        import tempfile
        from pathlib import Path

        stack = AffineStack()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "stack.json"

            stack.write(filepath)
            loaded = AffineStack.read(filepath)

        self.assertEqual(len(loaded), 0)
        self.assertIsNone(loaded.ndim)

    def test_get_metadata(self):
        print('Testing AffineStack: get_metadata ...')

        stack = AffineStack(
            metadata={
                "foo": 1,
                "bar": [1, 2, 3],
            }
        )

        self.assertEqual(
            stack.get_metadata("foo"),
            1,
        )

        self.assertEqual(
            stack.get_metadata("missing", 42),
            42,
        )

    def test_get_metadata_returns_copy(self):
        print('Testing AffineStack: get_metadata returns copy ...')

        stack = AffineStack(
            metadata={
                "foo": [1, 2, 3],
            }
        )

        value = stack.get_metadata("foo")
        value[0] = 999

        self.assertEqual(
            stack.get_metadata("foo"),
            [1, 2, 3],
        )

    def test_set_metadata(self):
        print('Testing AffineStack: set_metadata ...')

        stack = AffineStack()

        stack.set_metadata(
            "foo",
            [1, 2, 3],
        )

        self.assertEqual(
            stack.metadata,
            {"foo": [1, 2, 3]},
        )

    def test_has_metadata(self):
        print('Testing AffineStack: has_metadata ...')

        stack = AffineStack(
            metadata={
                "foo": 1,
            }
        )

        self.assertTrue(
            stack.has_metadata("foo")
        )

        self.assertFalse(
            stack.has_metadata("bar")
        )

    def test_interpolate_identity_scale(self):
        print('Testing AffineStack: interpolate identity scale ...')
        stack = AffineStack.from_array([
            [1., 0., 0., 0., 1., 0.],
            [1., 0., 2., 0., 1., 4.],
        ])

        result = stack.interpolate(1)

        self.assertEqual(len(result), 2)

        np.testing.assert_allclose(
            result.as_array(
                homogeneous=False,
                flatten=True,
            ),
            stack.as_array(
                homogeneous=False,
                flatten=True,
            ),
        )

    def test_interpolate_downsample(self):
        print('Testing AffineStack: interpolate downsample ...')
        stack = AffineStack.from_array([
            [1., 0., 0., 0., 1., 0.],
            [1., 0., 1., 0., 1., 2.],
            [1., 0., 2., 0., 1., 4.],
            [1., 0., 3., 0., 1., 6.],
        ])

        result = stack.interpolate(0.5)

        self.assertEqual(len(result), 2)

        np.testing.assert_allclose(
            result[0].translation,
            [0., 0.],
        )
        np.testing.assert_allclose(
            result[1].translation,
            [2., 4.],
        )

    def test_interpolate_preserves_properties(self):
        print('Testing AffineStack: interpolate preserves properties ...')
        stack = AffineStack.identity(
            length=4,
            ndim=2,
            pivot=[10., 20.],
            sequenced=True,
            metadata={"source": "test"},
            dtype=np.float32,
        )

        result = stack.interpolate(0.5)

        self.assertTrue(result.sequenced)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.metadata, {"source": "test"})

        np.testing.assert_allclose(
            result.pivot,
            [10., 20.],
        )

    def test_interpolate_rejects_invalid_scale(self):
        print('Testing AffineStack: interpolate rejects invalid scale ...')
        stack = AffineStack.identity(
            length=2,
            ndim=2,
        )

        with self.assertRaises(ValueError):
            stack.interpolate(0)

    def test_scaled_for_stack_resize(self):
        print('Testing AffineStack: scaled_for_stack_resize ...')
        stack = AffineStack(
            matrices=[
                AffineMatrix.from_translation(
                    [10., 20.],
                    pivot=[5., 6.],
                ),
                AffineMatrix.from_translation(
                    [20., 40.],
                    pivot=[5., 6.],
                ),
            ],
            sequenced=True,
        )

        result = stack.scaled_for_stack_resize(2)

        np.testing.assert_allclose(
            result.pivot,
            [10., 12.],
        )

        np.testing.assert_allclose(
            result[0].translation,
            [20., 40.],
        )

    def test_scaled_for_stack_resize_rejects_relative_stack(self):
        stack = AffineStack.identity(
            length=2,
            ndim=2,
            sequenced=False,
        )

        with self.assertRaisesRegex(
            ValueError,
            "requires a sequenced stack",
        ):
            stack.scaled_for_stack_resize(2)

    def test_matmul_with_matrix(self):
        print('Testing AffineStack: matmul with AffineMatrix ...')
        stack = AffineStack(
            matrices=[
                AffineMatrix.from_translation([1., 2.]),
                AffineMatrix.from_translation([3., 4.]),
            ]
        )

        offset = AffineMatrix.from_translation([10., 20.])

        result = stack @ offset

        np.testing.assert_allclose(
            result[0].translation,
            [11., 22.],
        )
        np.testing.assert_allclose(
            result[1].translation,
            [13., 24.],
        )

    def test_matmul_with_stack(self):
        print('Testing AffineStack: matmul with AffineStack ...')
        first = AffineStack(
            matrices=[
                AffineMatrix.from_translation([1., 2.]),
                AffineMatrix.from_translation([3., 4.]),
            ]
        )

        second = AffineStack(
            matrices=[
                AffineMatrix.from_translation([10., 20.]),
                AffineMatrix.from_translation([30., 40.]),
            ]
        )

        result = first @ second

        np.testing.assert_allclose(
            result[0].translation,
            [11., 22.],
        )
        np.testing.assert_allclose(
            result[1].translation,
            [33., 44.],
        )

    def test_matmul_rejects_different_stack_lengths(self):
        print('Testing AffineStack: matmul rejects different stack lengths ...')
        first = AffineStack.identity(2, ndim=2)
        second = AffineStack.identity(3, ndim=2)

        with self.assertRaisesRegex(
            ValueError,
            "same length",
        ):
            first @ second
    def test_equal(self):
        first = AffineStack.identity(
            length=2,
            ndim=2,
            sequenced=True,
            metadata={"source": "test"},
        )

        second = first.copy()

        self.assertEqual(first, second)

    def test_not_equal_transform(self):
        print('Testing AffineStack: not equal transform ...')
        first = AffineStack.identity(
            length=2,
            ndim=2,
        )

        second = AffineStack(
            matrices=[
                AffineMatrix.identity(2),
                AffineMatrix.from_translation([1., 0.]),
            ]
        )

        self.assertNotEqual(first, second)

    def test_array(self):
        print('Testing AffineStack: array ...')
        stack = AffineStack.identity(
            length=2,
            ndim=2,
        )

        result = np.asarray(stack)

        self.assertEqual(result.shape, (2, 3, 3))

        np.testing.assert_allclose(
            result,
            stack.as_homogeneous(),
        )

    def test_array_dtype(self):
        print('Testing AffineStack: array with dtype ...')
        stack = AffineStack.identity(
            length=2,
            ndim=2,
        )

        result = np.asarray(
            stack,
            dtype=np.float32,
        )

        self.assertEqual(result.dtype, np.float32)
        
    def test_composed_with_matrix(self):
        print('Testing AffineStack: composed_with AffineMatrix ...')
        stack = AffineStack(
            matrices=[
                AffineMatrix.from_translation([1., 2.]),
                AffineMatrix.from_translation([3., 4.]),
            ],
            sequenced=True,
            metadata={"source": "test"},
        )

        other = AffineMatrix.from_translation([10., 20.])

        result = stack.composed_with(other)

        np.testing.assert_allclose(
            result[0].translation,
            [11., 22.],
        )
        np.testing.assert_allclose(
            result[1].translation,
            [13., 24.],
        )

        self.assertTrue(result.sequenced)
        self.assertEqual(result.metadata, stack.metadata)

    def test_composed_with_stack(self):
        print('Testing AffineStack: composed_with AffineStack ...')
        first = AffineStack(
            matrices=[
                AffineMatrix.from_translation([1., 2.]),
                AffineMatrix.from_translation([3., 4.]),
            ],
            sequenced=True,
        )

        second = AffineStack(
            matrices=[
                AffineMatrix.from_translation([10., 20.]),
                AffineMatrix.from_translation([30., 40.]),
            ],
            sequenced=True,
        )

        result = first.composed_with(second)

        np.testing.assert_allclose(
            result[0].translation,
            [11., 22.],
        )
        np.testing.assert_allclose(
            result[1].translation,
            [33., 44.],
        )

    def test_inverse(self):
        print('Testing AffineStack: inverse ...')
        stack = AffineStack(
            matrices=[
                AffineMatrix.from_translation([1., 2.]),
                AffineMatrix.from_translation([3., 4.]),
            ],
            sequenced=True,
            metadata={"source": "test"},
        )

        result = stack.inverse()

        np.testing.assert_allclose(
            result[0].translation,
            [-1., -2.],
        )
        np.testing.assert_allclose(
            result[1].translation,
            [-3., -4.],
        )

        self.assertTrue(result.sequenced)
        self.assertEqual(result.metadata, stack.metadata)

    def test_inverse_composition_is_identity(self):
        print('Testing AffineStack: inverse composition is identity ...')
        stack = AffineStack(
            matrices=[
                AffineMatrix.from_array(
                    [
                        [2., 0., 10.],
                        [0., 3., 20.],
                    ]
                ),
                AffineMatrix.from_rotation(np.pi / 4),
            ],
            sequenced=True,
        )

        identity = stack @ stack.inverse()

        for matrix in identity:
            np.testing.assert_allclose(
                matrix.as_homogeneous(),
                np.eye(3),
                atol=1e-12,
            )

    def test_with_pivot(self):
        print('Testing AffineStack: with_pivot ...')
        stack = AffineStack.identity(
            length=2,
            ndim=2,
            pivot=[0., 0.],
            sequenced=True,
        )

        result = stack.with_pivot([10., 20.])

        np.testing.assert_allclose(
            result.pivot,
            [10., 20.],
        )

        for matrix in result:
            np.testing.assert_allclose(
                matrix.pivot,
                [10., 20.],
            )

        np.testing.assert_allclose(
            stack.pivot,
            [0., 0.],
        )

    def test_with_translations(self):
        print('Testing AffineStack: with_translations ...')
        stack = AffineStack.identity(
            length=2,
            ndim=2,
            sequenced=True,
        )

        result = stack.with_translations(
            [
                [10., 20.],
                [30., 40.],
            ]
        )

        np.testing.assert_allclose(
            result[0].translation,
            [10., 20.],
        )
        np.testing.assert_allclose(
            result[1].translation,
            [30., 40.],
        )

        np.testing.assert_allclose(
            stack[0].translation,
            [0., 0.],
        )

    def test_with_translations_rejects_invalid_shape(self):
        print('Testing AffineStack: with_translations rejects invalid shape ...')
        stack = AffineStack.identity(
            length=2,
            ndim=2,
        )

        with self.assertRaisesRegex(
            ValueError,
            "translations must have shape",
        ):
            stack.with_translations(
                [[1., 2.]]
            )

    def test_add_to_translations_single_offset(self):
        print('Testing AffineStack: add_to_translations single offset ...')
        stack = AffineStack(
            matrices=[
                AffineMatrix.from_translation([1., 2.]),
                AffineMatrix.from_translation([3., 4.]),
            ]
        )

        result = stack.add_to_translations([10., 20.])

        np.testing.assert_allclose(
            result[0].translation,
            [11., 22.],
        )
        np.testing.assert_allclose(
            result[1].translation,
            [13., 24.],
        )

    def test_add_to_translations_per_matrix_offsets(self):
        print('Testing AffineStack: add_to_translations per-matrix offsets ...')
        stack = AffineStack(
            matrices=[
                AffineMatrix.from_translation([1., 2.]),
                AffineMatrix.from_translation([3., 4.]),
            ]
        )

        result = stack.add_to_translations(
            [
                [10., 20.],
                [30., 40.],
            ]
        )

        np.testing.assert_allclose(
            result[0].translation,
            [11., 22.],
        )
        np.testing.assert_allclose(
            result[1].translation,
            [33., 44.],
        )

    def test_add_to_translations_rejects_invalid_shape(self):
        print('Testing AffineStack: add_to_translations rejects invalid shape ...')
        stack = AffineStack.identity(
            length=2,
            ndim=2,
        )

        with self.assertRaisesRegex(
            ValueError,
            "offsets must have shape",
        ):
            stack.add_to_translations(
                [1., 2., 3.]
            )

    def test_get_substack(self):
        print('Testing AffineStack: get_substack ...')
        stack = AffineStack(
            matrices=[
                AffineMatrix.from_translation([0., 0.]),
                AffineMatrix.from_translation([1., 2.]),
                AffineMatrix.from_translation([3., 4.]),
                AffineMatrix.from_translation([5., 6.]),
            ],
            sequenced=True,
            metadata={
                "bounds": [
                    [0, 0, 10, 10],
                    [1, 1, 10, 10],
                    [2, 2, 10, 10],
                    [3, 3, 10, 10],
                ],
                "stack_shape": [4, 100, 200],
                "source": "test",
            },
        )

        result = stack.get_substack(slice(1, 3))

        self.assertEqual(len(result), 2)
        self.assertTrue(result.sequenced)

        np.testing.assert_allclose(
            result[0].translation,
            [1., 2.],
        )
        np.testing.assert_allclose(
            result[1].translation,
            [3., 4.],
        )

        self.assertEqual(
            result.get_metadata("bounds"),
            [
                [1, 1, 10, 10],
                [2, 2, 10, 10],
            ],
        )
        self.assertEqual(
            result.get_metadata("stack_shape"),
            [2, 100, 200],
        )
        self.assertEqual(
            result.get_metadata("source"),
            "test",
        )

    def test_get_substack_numpy_metadata(self):
        print('Testing AffineStack: get_substack numpy metadata ...')
        stack = AffineStack.identity(
            length=4,
            ndim=2,
            metadata={
                "bounds": np.array(
                    [
                        [0, 0, 10, 10],
                        [1, 1, 10, 10],
                        [2, 2, 10, 10],
                        [3, 3, 10, 10],
                    ]
                ),
            },
        )

        result = stack.get_substack(slice(1, 3))

        np.testing.assert_array_equal(
            result.get_metadata("bounds"),
            np.array(
                [
                    [1, 1, 10, 10],
                    [2, 2, 10, 10],
                ]
            ),
        )

    def test_get_substack_rejects_non_slice(self):
        print('Testing AffineStack: get_substack rejects non-slice ...')
        stack = AffineStack.identity(
            length=3,
            ndim=2,
        )

        with self.assertRaisesRegex(
            TypeError,
            "selection must be a slice",
        ):
            stack.get_substack([0, 1])

    def test_getitem_slice_uses_substack_metadata(self):
        print('Testing AffineStack: getitem slice uses substack metadata ...')
        stack = AffineStack.identity(
            length=4,
            ndim=2,
            metadata={
                "bounds": [
                    [0, 0, 10, 10],
                    [1, 1, 10, 10],
                    [2, 2, 10, 10],
                    [3, 3, 10, 10],
                ],
                "stack_shape": [4, 100, 200],
            },
        )

        result = stack[1:3]

        self.assertEqual(
            result.get_metadata("bounds"),
            [
                [1, 1, 10, 10],
                [2, 2, 10, 10],
            ],
        )
        self.assertEqual(
            result.get_metadata("stack_shape"),
            [2, 100, 200],
        )

    def test_smooth_gaussian_zero_returns_copy(self):
        stack = AffineStack.identity(
            length=3,
            ndim=2,
            sequenced=True,
        )

        result = stack.smooth_gaussian(0)

        self.assertIsNot(result, stack)
        self.assertEqual(result, stack)
        
    def test_smooth_gaussian_reduces_translation_peak(self):
        stack = AffineStack(
            matrices=[
                AffineMatrix.from_translation([0., 0.]),
                AffineMatrix.from_translation([0., 0.]),
                AffineMatrix.from_translation([10., 20.]),
                AffineMatrix.from_translation([0., 0.]),
                AffineMatrix.from_translation([0., 0.]),
            ],
            sequenced=True,
            metadata={"source": "test"},
        )

        result = stack.smooth_gaussian(1)

        self.assertLess(result[2].translation[0], 10.)
        self.assertGreater(result[1].translation[0], 0.)
        self.assertGreater(result[3].translation[0], 0.)
        self.assertTrue(result.sequenced)
        self.assertEqual(result.metadata, stack.metadata)

    def test_smooth_gaussian_rejects_negative_sigma(self):
        stack = AffineStack.identity(
            length=3,
            ndim=2,
        )

        with self.assertRaisesRegex(
            ValueError,
            "non-negative",
        ):
            stack.smooth_gaussian(-1)

    def test_smooth_median_zero_returns_copy(self):
        stack = AffineStack.identity(
            length=3,
            ndim=2,
        )

        result = stack.smooth_median(0)

        self.assertIsNot(result, stack)
        self.assertEqual(result, stack)

    def test_smooth_median_removes_translation_outlier(self):
        stack = AffineStack(
            matrices=[
                AffineMatrix.from_translation([1., 2.]),
                AffineMatrix.from_translation([1., 2.]),
                AffineMatrix.from_translation([100., 200.]),
                AffineMatrix.from_translation([1., 2.]),
                AffineMatrix.from_translation([1., 2.]),
            ]
        )

        result = stack.smooth_median(radius=1)

        np.testing.assert_allclose(
            result[2].translation,
            [1., 2.],
        )
    def test_smooth_median_rejects_non_integer_radius(self):
        stack = AffineStack.identity(
            length=3,
            ndim=2,
        )

        with self.assertRaisesRegex(
            TypeError,
            "radius must be an integer",
        ):
            stack.smooth_median(1.5)

    def test_smooth_median_rejects_negative_radius(self):

        stack = AffineStack.identity(
            length=3,
            ndim=2,
        )

        with self.assertRaisesRegex(
            ValueError,
            "non-negative",
        ):
            stack.smooth_median(-1)

    def test_replace_large_translations_default(self):

        stack = AffineStack(
            matrices=[
                AffineMatrix.from_translation([1., 2.]),
                AffineMatrix.from_translation([100., 200.]),
                AffineMatrix.from_translation([3., 4.]),
            ],
            sequenced=True,
        )

        result = stack.replace_large_translations(10.)

        np.testing.assert_allclose(
            result[0].translation,
            [1., 2.],
        )
        np.testing.assert_allclose(
            result[1].translation,
            [0., 0.],
        )
        np.testing.assert_allclose(
            result[2].translation,
            [3., 4.],
        )

    def test_replace_large_translations_custom_replacement(self):

        stack = AffineStack(
            matrices=[
                AffineMatrix.from_translation([100., 200.]),
            ]
        )

        replacement = AffineMatrix.from_translation([-1., -2.])

        result = stack.replace_large_translations(
            10.,
            replacement=replacement,
        )

        np.testing.assert_allclose(
            result[0].translation,
            [-1., -2.],
        )

    def test_replace_large_translations_preserves_metadata(self):

        stack = AffineStack.identity(
            length=2,
            ndim=2,
            sequenced=True,
            metadata={"source": "test"},
        )

        result = stack.replace_large_translations(1.)

        self.assertTrue(result.sequenced)
        self.assertEqual(result.metadata, stack.metadata)

    def test_replace_large_translations_rejects_negative_distance(self):
        
        stack = AffineStack.identity(
            length=1,
            ndim=2,
        )

        with self.assertRaisesRegex(
            ValueError,
            "non-negative",
        ):
            stack.replace_large_translations(-1.)

class TestAffineStackApplyZStep(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_apply_z_step(self):
        stack = AffineStack(
            matrices=[
                AffineMatrix.from_translation([0., 0.]),
                AffineMatrix.from_translation([2., 4.]),
                AffineMatrix.from_translation([4., 8.]),
            ],
            sequenced=True,
            metadata={"z_step": 2},
        )

        result = stack.apply_z_step()

        self.assertTrue(result.sequenced)
        self.assertEqual(result.get_metadata("z_step"), 1)
        self.assertEqual(len(result), 6)

        np.testing.assert_allclose(
            result[0].translation,
            [0., 0.],
        )
        np.testing.assert_allclose(
            result[1].translation,
            [1., 2.],
        )
        np.testing.assert_allclose(
            result[2].translation,
            [2., 4.],
        )
        np.testing.assert_allclose(
            result[3].translation,
            [3., 6.],
        )
        np.testing.assert_allclose(
            result[4].translation,
            [4., 8.],
        )

    def test_apply_z_step_preserves_original(self):
        stack = AffineStack(
            matrices=[
                AffineMatrix.from_translation([0., 0.]),
                AffineMatrix.from_translation([2., 4.]),
            ],
            sequenced=True,
            metadata={"z_step": 2},
        )

        result = stack.apply_z_step()

        self.assertEqual(len(stack), 2)
        self.assertEqual(stack.get_metadata("z_step"), 2)
        self.assertIsNot(result, stack)

    def test_apply_z_step_one_returns_copy(self):
        stack = AffineStack.identity(
            length=2,
            ndim=2,
            sequenced=True,
            metadata={"z_step": 1},
        )

        result = stack.apply_z_step()

        self.assertIsNot(result, stack)
        self.assertEqual(len(result), len(stack))
        self.assertEqual(result.get_metadata("z_step"), 1)

        for original, copied in zip(stack, result):
            self.assertEqual(original, copied)

    def test_apply_z_step_preserves_pivot_and_dtype(self):
        stack = AffineStack(
            matrices=[
                AffineMatrix.from_translation(
                    [0., 0.],
                    pivot=[10., 20.],
                    dtype=np.float32,
                ),
                AffineMatrix.from_translation(
                    [2., 4.],
                    pivot=[10., 20.],
                    dtype=np.float32,
                ),
            ],
            sequenced=True,
            metadata={"z_step": 2},
        )

        result = stack.apply_z_step()

        self.assertEqual(result.dtype, np.float32)

        np.testing.assert_allclose(
            result.pivot,
            [10., 20.],
        )

    def test_apply_z_step_requires_metadata(self):
        stack = AffineStack.identity(
            length=2,
            ndim=2,
            sequenced=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "no 'z_step' entry",
        ):
            stack.apply_z_step()

    def test_apply_z_step_requires_sequenced_stack(self):
        stack = AffineStack.identity(
            length=2,
            ndim=2,
            sequenced=False,
            metadata={"z_step": 2},
        )

        with self.assertRaisesRegex(
            ValueError,
            "requires a sequenced",
        ):
            stack.apply_z_step()

    def test_apply_z_step_rejects_invalid_z_step(self):
        stack = AffineStack.identity(
            length=2,
            ndim=2,
            sequenced=True,
            metadata={"z_step": 0},
        )

        with self.assertRaisesRegex(
            ValueError,
            "greater than zero",
        ):
            stack.apply_z_step()

    def test_apply_z_step_requires_two_transforms(self):
        stack = AffineStack.identity(
            length=1,
            ndim=2,
            sequenced=True,
            metadata={"z_step": 2},
        )

        with self.assertRaisesRegex(
            ValueError,
            "at least two transforms",
        ):
            stack.apply_z_step()
