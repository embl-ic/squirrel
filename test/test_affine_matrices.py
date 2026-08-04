import unittest
import warnings
import numpy as np

from squirrel.library.affine_matrices import AffineMatrix


class TestAffineMatrix(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_identity_2d(self):
        print(f'Testing AffineMatrix: identity 2D ...')
        affine = AffineMatrix.identity(ndim=2)

        np.testing.assert_allclose(
            affine._matrix,
            np.eye(3),
        )
        self.assertEqual(affine._ndim, 2)
        np.testing.assert_allclose(
            affine._pivot,
            [0.0, 0.0],
        )

    def test_identity_3d(self):
        print(f'Testing AffineMatrix: identity 3D ...')
        affine = AffineMatrix.identity(ndim=3)

        np.testing.assert_allclose(
            affine._matrix,
            np.eye(4),
        )
        self.assertEqual(affine._ndim, 3)
        np.testing.assert_allclose(
            affine._pivot,
            [0.0, 0.0, 0.0],
        )

    def test_identity_rejects_invalid_dimension(self):
        print(f'Testing AffineMatrix: identity rejects invalid dimension ...')
        with self.assertRaisesRegex(
            ValueError,
            "Only 2D and 3D",
        ):
            AffineMatrix.identity(ndim=4)

    def test_from_array_accepts_compact_2d_matrix(self):
        print(f'Testing AffineMatrix: from array accepts compact 2D matrix ...')
        affine = AffineMatrix.from_array(
            [
                [1.0, 0.0, 10.0],
                [0.0, 1.0, 20.0],
            ]
        )

        expected = np.array(
            [
                [1.0, 0.0, 10.0],
                [0.0, 1.0, 20.0],
                [0.0, 0.0, 1.0],
            ]
        )

        np.testing.assert_allclose(
            affine._matrix,
            expected,
        )

    def test_from_array_accepts_flat_compact_2d(self):
        print(f'Testing AffineMatrix: from array accepts flat compact 2D ...')
        affine = AffineMatrix.from_array(
            [1.0, 0.0, 10.0, 0.0, 1.0, 20.0]
        )

        expected = np.array(
            [
                [1.0, 0.0, 10.0],
                [0.0, 1.0, 20.0],
                [0.0, 0.0, 1.0],
            ]
        )

        np.testing.assert_allclose(
            affine._matrix,
            expected,
        )

    def test_from_array_accepts_homogeneous_2d(self):
        print(f'Testing AffineMatrix: from array accepts homogeneous 2D ...')
        input_matrix = np.array(
            [
                [2.0, 0.0, 5.0],
                [0.0, 3.0, 6.0],
                [0.0, 0.0, 1.0],
            ]
        )

        affine = AffineMatrix.from_array(input_matrix)

        np.testing.assert_allclose(
            affine._matrix,
            input_matrix,
        )

    def test_from_array_accepts_compact_3d_matrix(self):
        print(f'Testing AffineMatrix: from array accepts compact 3D matrix ...')
        input_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 20.0],
                [0.0, 0.0, 1.0, 30.0],
            ]
        )

        affine = AffineMatrix.from_array(input_matrix)

        expected = np.eye(4)
        expected[:3, :] = input_matrix

        np.testing.assert_allclose(
            affine._matrix,
            expected,
        )

    def test_from_array_rejects_invalid_shape(self):
        print(f'Testing AffineMatrix: from array rejects invalid shape ...')
        with self.assertRaisesRegex(
            ValueError,
            "Invalid affine matrix shape",
        ):
            AffineMatrix.from_array(np.eye(5))

    def test_from_array_rejects_invalid_homogeneous_row(self):
        print(f'Testing AffineMatrix: from array rejects invalid homogeneous row ...')
        matrix = np.array(
            [
                [1.0, 0.0, 10.0],
                [0.0, 1.0, 20.0],
                [0.0, 1.0, 1.0],
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "final row",
        ):
            AffineMatrix.from_array(matrix)

    def test_from_array_copies_input(self):
        print(f'Testing AffineMatrix: from array copies input ...')
        input_matrix = np.eye(3)
        affine = AffineMatrix.from_array(input_matrix)

        input_matrix[0, 0] = 10.0

        self.assertEqual(
            affine._matrix[0, 0],
            1.0,
        )

    def test_from_translation_2d(self):
        print(f'Testing AffineMatrix: from translation 2D ...')
        affine = AffineMatrix.from_translation(
            [10.0, 20.0]
        )

        expected = np.array(
            [
                [1.0, 0.0, 10.0],
                [0.0, 1.0, 20.0],
                [0.0, 0.0, 1.0],
            ]
        )

        np.testing.assert_allclose(
            affine._matrix,
            expected,
        )

    def test_from_translation_3d(self):
        print(f'Testing AffineMatrix: from translation 3D ...')
        affine = AffineMatrix.from_translation(
            [10.0, 20.0, 30.0]
        )

        expected = np.eye(4)
        expected[:3, 3] = [10.0, 20.0, 30.0]

        np.testing.assert_allclose(
            affine._matrix,
            expected,
        )

    def test_from_translation_rejects_scalar(self):
        print(f'Testing AffineMatrix: from translation rejects scalar ...')
        with self.assertRaisesRegex(
            ValueError,
            "one-dimensional",
        ):
            AffineMatrix.from_translation(10.0)

    def test_from_rotation_2d_zero(self):
        affine = AffineMatrix.from_rotation(0.0)

        np.testing.assert_allclose(
            affine._matrix,
            np.eye(3),
            atol=1e-12,
        )

    def test_from_rotation_2d_quarter_turn(self):
        print(f'Testing AffineMatrix: from rotation 2D quarter turn ...')
        affine = AffineMatrix.from_rotation(
            np.pi / 2
        )

        expected = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        np.testing.assert_allclose(
            affine._matrix,
            expected,
            atol=1e-12,
        )

    def test_from_rotation_accepts_2d_rotation_matrix(self):
        print(f'Testing AffineMatrix: from rotation accepts 2D rotation matrix ...')
        rotation = np.array(
            [
                [0.0, -1.0],
                [1.0, 0.0],
            ]
        )

        affine = AffineMatrix.from_rotation(rotation)

        np.testing.assert_allclose(
            affine._matrix[:2, :2],
            rotation,
        )

    def test_from_rotation_accepts_3d_rotation_matrix(self):
        print(f'Testing AffineMatrix: from rotation accepts 3D rotation matrix ...')
        affine = AffineMatrix.from_rotation(np.eye(3))

        np.testing.assert_allclose(
            affine._matrix,
            np.eye(4),
        )

    def test_from_rotation_accepts_zero_euler_angles(self):
        print(f'Testing AffineMatrix: from rotation accepts zero Euler angles ...')
        affine = AffineMatrix.from_rotation(
            [0.0, 0.0, 0.0]
        )

        np.testing.assert_allclose(
            affine._matrix,
            np.eye(4),
            atol=1e-12,
        )

    def test_from_rotation_rejects_invalid_shape(self):
        print(f'Testing AffineMatrix: from rotation rejects invalid shape ...')
        with self.assertRaisesRegex(
            ValueError,
            "Rotation must be",
        ):
            AffineMatrix.from_rotation([1.0, 2.0])

    def test_from_scale_uniform_2d(self):
        print(f'Testing AffineMatrix: from scale uniform 2D ...')
        affine = AffineMatrix.from_scale(
            2.0,
            ndim=2,
        )

        expected = np.diag([2.0, 2.0, 1.0])

        np.testing.assert_allclose(
            affine._matrix,
            expected,
        )

    def test_from_scale_nonuniform_2d(self):
        print(f'Testing AffineMatrix: from scale nonuniform 2D ...')
        affine = AffineMatrix.from_scale(
            [2.0, 3.0]
        )

        expected = np.diag([2.0, 3.0, 1.0])

        np.testing.assert_allclose(
            affine._matrix,
            expected,
        )

    def test_from_scale_nonuniform_3d(self):
        print(f'Testing AffineMatrix: from scale nonuniform 3D ...')
        affine = AffineMatrix.from_scale(
            [2.0, 3.0, 4.0]
        )

        expected = np.diag([2.0, 3.0, 4.0, 1.0])

        np.testing.assert_allclose(
            affine._matrix,
            expected,
        )

    def test_from_scale_scalar_requires_ndim(self):
        print(f'Testing AffineMatrix: from scale scalar requires ndim ...')
        with self.assertRaisesRegex(
            ValueError,
            "ndim must be supplied",
        ):
            AffineMatrix.from_scale(2.0)

    def test_constructor_accepts_pivot(self):
        print(f'Testing AffineMatrix: constructor accepts pivot ...')
        affine = AffineMatrix.identity(
            ndim=2,
            pivot=[100.0, 200.0],
        )

        np.testing.assert_allclose(
            affine._pivot,
            [100.0, 200.0],
        )

    def test_constructor_rejects_incorrect_pivot_size(self):
        print(f'Testing AffineMatrix: constructor rejects incorrect pivot size ...')
        with self.assertRaisesRegex(
            ValueError,
            "Pivot must have shape",
        ):
            AffineMatrix.identity(
                ndim=2,
                pivot=[1.0, 2.0, 3.0],
            )

    def test_ndim_property(self):
        print(f'Testing AffineMatrix: ndim property ...')
        affine = AffineMatrix.identity(2)
        self.assertEqual(affine.ndim, 2)


    def test_pivot_property(self):
        print(f'Testing AffineMatrix: pivot property ...')
        affine = AffineMatrix.identity(2, pivot=[1, 2])

        np.testing.assert_array_equal(
            affine.pivot,
            [1, 2]
        )


    def test_pivot_returns_copy(self):
        print(f'Testing AffineMatrix: pivot returns copy ...')
        affine = AffineMatrix.identity(2, pivot=[1, 2])

        pivot = affine.pivot
        pivot[0] = 100

        np.testing.assert_array_equal(
            affine.pivot,
            [1, 2]
        )


    def test_dtype_property(self):
        print(f'Testing AffineMatrix: dtype property ...')
        affine = AffineMatrix.identity(2, dtype=np.float32)

        self.assertEqual(affine.dtype, np.float32)


    def test_linear_property(self):
        print(f'Testing AffineMatrix: linear property ...')
        affine = AffineMatrix.from_array(
            [[1, 2, 3],
            [4, 5, 6]]
        )

        np.testing.assert_array_equal(
            affine.linear,
            [[1, 2],
            [4, 5]]
        )


    def test_translation_property(self):
        print(f'Testing AffineMatrix: translation property ...')
        affine = AffineMatrix.from_translation([10, 20])

        np.testing.assert_array_equal(
            affine.translation,
            [10, 20]
        )


    def test_linear_returns_copy(self):
        print(f'Testing AffineMatrix: linear returns copy ...')
        affine = AffineMatrix.identity(2)

        linear = affine.linear
        linear[0, 0] = 100

        self.assertEqual(
            affine.linear[0, 0],
            1
        )


    def test_translation_returns_copy(self):
        print(f'Testing AffineMatrix: translation returns copy ...')
        affine = AffineMatrix.from_translation([10, 20])

        t = affine.translation
        t[0] = 999

        np.testing.assert_array_equal(
            affine.translation,
            [10, 20]
        )


class TestAffineMatrixTransformations(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_compose_translations(self):
        print(f'Testing AffineMatrix: compose translations ...')
        first = AffineMatrix.from_translation([10.0, 20.0])
        second = AffineMatrix.from_translation([1.0, 2.0])

        result = first.compose(second)

        np.testing.assert_allclose(
            result.translation,
            [11.0, 22.0],
        )

    def test_compose_applies_other_first(self):
        print(f'Testing AffineMatrix: compose applies other first ...')
        rotation = AffineMatrix.from_rotation(np.pi / 2)
        translation = AffineMatrix.from_translation([1.0, 0.0])

        result = rotation.compose(translation)

        transformed = result.apply([0.0, 0.0])

        np.testing.assert_allclose(
            transformed,
            [0.0, 1.0],
            atol=1e-12,
        )

    def test_compose_rejects_different_dimensions(self):
        print(f'Testing AffineMatrix: compose rejects different dimensions ...')
        affine_2d = AffineMatrix.identity(2)
        affine_3d = AffineMatrix.identity(3)

        with self.assertRaisesRegex(
            ValueError,
            "different dimensions",
        ):
            affine_2d.compose(affine_3d)

    def test_compose_rejects_different_pivots(self):
        print(f'Testing AffineMatrix: compose rejects different pivots ...')
        first = AffineMatrix.identity(
            2,
            pivot=[0.0, 0.0],
        )
        second = AffineMatrix.identity(
            2,
            pivot=[1.0, 0.0],
        )

        with self.assertRaisesRegex(
            ValueError,
            "different pivots",
        ):
            first.compose(second)

    def test_inverse(self):
        print(f'Testing AffineMatrix: inverse ...')
        affine = AffineMatrix.from_array(
            [
                [2.0, 0.0, 10.0],
                [0.0, 3.0, 20.0],
            ]
        )

        identity = affine.compose(affine.inverse())

        np.testing.assert_allclose(
            identity._matrix,
            np.eye(3),
            atol=1e-12,
        )

    def test_inverse_rejects_singular_transform(self):
        print(f'Testing AffineMatrix: inverse rejects singular transform ...')
        affine = AffineMatrix.from_array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "singular",
        ):
            affine.inverse()

    def test_apply_single_point(self):
        print(f'Testing AffineMatrix: apply single point ...')
        affine = AffineMatrix.from_translation(
            [10.0, 20.0]
        )

        result = affine.apply([1.0, 2.0])

        np.testing.assert_allclose(
            result,
            [11.0, 22.0],
        )

    def test_apply_multiple_points(self):
        print(f'Testing AffineMatrix: apply multiple points ...')
        affine = AffineMatrix.from_translation(
            [10.0, 20.0]
        )

        points = np.array(
            [
                [0.0, 0.0],
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )

        result = affine.apply(points)

        expected = np.array(
            [
                [10.0, 20.0],
                [11.0, 22.0],
                [13.0, 24.0],
            ]
        )

        np.testing.assert_allclose(result, expected)

    def test_apply_rotation_around_pivot(self):
        print(f'Testing AffineMatrix: apply rotation around pivot ...')
        affine = AffineMatrix.from_rotation(
            np.pi / 2,
            pivot=[1.0, 1.0],
        )

        result = affine.apply([2.0, 1.0])

        np.testing.assert_allclose(
            result,
            [1.0, 2.0],
            atol=1e-12,
        )

    def test_apply_rejects_incorrect_point_dimension(self):
        print(f'Testing AffineMatrix: apply rejects incorrect point dimension ...')
        affine = AffineMatrix.identity(2)

        with self.assertRaisesRegex(
            ValueError,
            "last point dimension",
        ):
            affine.apply([1.0, 2.0, 3.0])

    def test_with_translation(self):
        print(f'Testing AffineMatrix: with translation ...')
        affine = AffineMatrix.identity(2)

        result = affine.with_translation([10.0, 20.0])

        np.testing.assert_allclose(
            result.translation,
            [10.0, 20.0],
        )
        np.testing.assert_allclose(
            affine.translation,
            [0.0, 0.0],
        )

    def test_with_pivot(self):
        print(f'Testing AffineMatrix: with pivot ...')
        affine = AffineMatrix.identity(2)

        result = affine.with_pivot([10.0, 20.0])

        np.testing.assert_allclose(
            result.pivot,
            [10.0, 20.0],
        )
        np.testing.assert_allclose(
            affine.pivot,
            [0.0, 0.0],
        )

    def test_shifted_pivot_to_origin_preserves_transform(self):
        print(f'Testing AffineMatrix: shifted pivot to origin preserves transform ...')
        affine = AffineMatrix.from_rotation(
            np.pi / 2,
            pivot=[10.0, 20.0],
        )

        shifted = affine.shifted_pivot_to_origin()

        points = np.array(
            [
                [0.0, 0.0],
                [10.0, 20.0],
                [15.0, 20.0],
            ]
        )

        np.testing.assert_allclose(
            shifted.apply(points),
            affine.apply(points),
            atol=1e-12,
        )

        np.testing.assert_allclose(
            shifted.pivot,
            [0.0, 0.0],
        )

    def test_scaled_for_uniform_image_resize(self):
        print(f'Testing AffineMatrix: scaled for uniform image resize ...')
        affine = AffineMatrix.from_translation(
            [10.0, 20.0],
            pivot=[5.0, 6.0],
        )

        scaled = affine.scaled_for_image_resize(2.0)

        np.testing.assert_allclose(
            scaled.translation,
            [20.0, 40.0],
        )
        np.testing.assert_allclose(
            scaled.pivot,
            [10.0, 12.0],
        )
        np.testing.assert_allclose(
            scaled.linear,
            affine.linear,
        )

    def test_scaled_for_image_resize_preserves_coordinate_relation(self):
        print(f'Testing AffineMatrix: scaled for image resize preserves coordinate relation ...')
        affine = AffineMatrix.from_rotation(
            np.pi / 4,
            pivot=[10.0, 20.0],
        ).with_translation([3.0, 4.0])

        scale = 2.0
        scaled = affine.scaled_for_image_resize(scale)

        point = np.array([5.0, 8.0])

        expected = scale * affine.apply(point)
        actual = scaled.apply(scale * point)

        np.testing.assert_allclose(
            actual,
            expected,
            atol=1e-12,
        )

    def test_to_3d_xy_plane(self):
        print(f'Testing AffineMatrix: to 3D xy plane ...')
        affine = AffineMatrix.from_translation(
            [10.0, 20.0],
            pivot=[1.0, 2.0],
        )

        result = affine.to_3d(axis=2)

        self.assertEqual(result.ndim, 3)

        np.testing.assert_allclose(
            result.translation,
            [10.0, 20.0, 0.0],
        )
        np.testing.assert_allclose(
            result.pivot,
            [1.0, 2.0, 0.0],
        )

    def test_to_3d_xz_plane(self):
        print(f'Testing AffineMatrix: to 3D xz plane ...')
        affine = AffineMatrix.from_translation(
            [10.0, 20.0],
            pivot=[1.0, 2.0],
        )

        result = affine.to_3d(axis=1)

        np.testing.assert_allclose(
            result.translation,
            [10.0, 0.0, 20.0],
        )
        np.testing.assert_allclose(
            result.pivot,
            [1.0, 0.0, 2.0],
        )

    def test_to_3d_yz_plane(self):
        print(f'Testing AffineMatrix: to 3D yz plane ...')
        affine = AffineMatrix.from_translation(
            [10.0, 20.0],
            pivot=[1.0, 2.0],
        )

        result = affine.to_3d(axis=0)

        np.testing.assert_allclose(
            result.translation,
            [0.0, 10.0, 20.0],
        )
        np.testing.assert_allclose(
            result.pivot,
            [0.0, 1.0, 2.0],
        )

    def test_to_3d_rejects_invalid_axis(self):
        print(f'Testing AffineMatrix: to 3D rejects invalid axis ...')
        affine = AffineMatrix.identity(2)

        with self.assertRaisesRegex(
            ValueError,
            "axis must be",
        ):
            affine.to_3d(axis=3)


class TestAffineMatrixDecomposition(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_decompose_2d_identity(self):
        print(f'Testing AffineMatrix: decompose 2D identity ...')
        affine = AffineMatrix.identity(2)

        translation, rotation, scale, shear = affine.decompose()

        np.testing.assert_allclose(
            translation._matrix,
            np.eye(3),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            rotation._matrix,
            np.eye(3),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            scale._matrix,
            np.eye(3),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            shear._matrix,
            np.eye(3),
            atol=1e-12,
        )

    def test_decompose_3d_identity(self):
        print(f'Testing AffineMatrix: decompose 3D identity ...')
        affine = AffineMatrix.identity(3)

        translation, rotation, scale, shear = affine.decompose()

        np.testing.assert_allclose(
            translation._matrix,
            np.eye(4),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            rotation._matrix,
            np.eye(4),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            scale._matrix,
            np.eye(4),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            shear._matrix,
            np.eye(4),
            atol=1e-12,
        )

    def test_decompose_2d_translation(self):
        print(f'Testing AffineMatrix: decompose 2D translation ...')
        affine = AffineMatrix.from_translation(
            [10.0, 20.0]
        )

        translation, rotation, scale, shear = affine.decompose()

        np.testing.assert_allclose(
            translation.translation,
            [10.0, 20.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            rotation.linear,
            np.eye(2),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            scale.linear,
            np.eye(2),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            shear.linear,
            np.eye(2),
            atol=1e-12,
        )

    def test_decompose_3d_translation(self):
        print(f'Testing AffineMatrix: decompose 3D translation ...')
        affine = AffineMatrix.from_translation(
            [10.0, 20.0, 30.0]
        )

        translation, rotation, scale, shear = affine.decompose()

        np.testing.assert_allclose(
            translation.translation,
            [10.0, 20.0, 30.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            rotation.linear,
            np.eye(3),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            scale.linear,
            np.eye(3),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            shear.linear,
            np.eye(3),
            atol=1e-12,
        )

    def test_decompose_2d_scale(self):
        print(f'Testing AffineMatrix: decompose 2D scale ...')
        affine = AffineMatrix.from_scale(
            [2.0, 3.0]
        )

        translation, rotation, scale, shear = affine.decompose()

        np.testing.assert_allclose(
            translation.translation,
            [0.0, 0.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            rotation.linear,
            np.eye(2),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            scale.linear,
            np.diag([2.0, 3.0]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            shear.linear,
            np.eye(2),
            atol=1e-12,
        )

    def test_decompose_3d_scale(self):
        print(f'Testing AffineMatrix: decompose 3D scale ...')
        affine = AffineMatrix.from_scale(
            [2.0, 3.0, 4.0]
        )

        translation, rotation, scale, shear = affine.decompose()

        np.testing.assert_allclose(
            scale.linear,
            np.diag([2.0, 3.0, 4.0]),
            atol=1e-12,
        )

    def test_decompose_2d_rotation(self):
        print(f'Testing AffineMatrix: decompose 2D rotation ...')
        affine = AffineMatrix.from_rotation(
            np.pi / 4
        )

        translation, rotation, scale, shear = affine.decompose()

        np.testing.assert_allclose(
            rotation.linear,
            affine.linear,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            scale.linear,
            np.eye(2),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            shear.linear,
            np.eye(2),
            atol=1e-12,
        )

    def test_decompose_3d_rotation(self):
        print(f'Testing AffineMatrix: decompose 3D rotation ...')
        affine = AffineMatrix.from_rotation(
            [0.2, -0.3, 0.4]
        )

        translation, rotation, scale, shear = affine.decompose()

        np.testing.assert_allclose(
            rotation.linear,
            affine.linear,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            scale.linear,
            np.eye(3),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            shear.linear,
            np.eye(3),
            atol=1e-12,
        )

    def test_decompose_2d_recomposition(self):
        print(f'Testing AffineMatrix: decompose 2D recomposition ...')
        affine = (
            AffineMatrix.from_translation([10.0, 20.0])
            .compose(AffineMatrix.from_rotation(np.pi / 4))
            .compose(AffineMatrix.from_scale([2.0, 3.0]))
        )

        translation, rotation, scale, shear = affine.decompose()

        recomposed = (
            translation
            .compose(rotation)
            .compose(scale)
            .compose(shear)
        )

        np.testing.assert_allclose(
            recomposed._matrix,
            affine._matrix,
            atol=1e-10,
        )

    def test_decompose_3d_recomposition(self):
        print(f'Testing AffineMatrix: decompose 3D recomposition ...')
        affine = (
            AffineMatrix.from_translation([10.0, 20.0, 30.0])
            .compose(AffineMatrix.from_rotation([0.2, -0.3, 0.4]))
            .compose(AffineMatrix.from_scale([2.0, 3.0, 4.0]))
        )

        translation, rotation, scale, shear = affine.decompose()

        recomposed = (
            translation
            .compose(rotation)
            .compose(scale)
            .compose(shear)
        )

        np.testing.assert_allclose(
            recomposed._matrix,
            affine._matrix,
            atol=1e-10,
        )

    def test_decompose_shifted_pivot_recomposition(self):
        print(f'Testing AffineMatrix: decompose shifted pivot recomposition ...')
        
        affine = AffineMatrix.from_rotation(
            np.pi / 3,
            pivot=[10.0, 20.0],
        ).with_translation([5.0, 7.0])

        translation, rotation, scale, shear = affine.decompose()

        recomposed = (
            translation
            .compose(rotation)
            .compose(scale)
            .compose(shear)
        )

        expected = affine.shifted_pivot_to_origin()

        np.testing.assert_allclose(
            recomposed._matrix,
            expected._matrix,
            atol=1e-10,
        )

        np.testing.assert_allclose(
            translation.pivot,
            [0.0, 0.0],
        )
        np.testing.assert_allclose(
            rotation.pivot,
            [0.0, 0.0],
        )
        np.testing.assert_allclose(
            scale.pivot,
            [0.0, 0.0],
        )
        np.testing.assert_allclose(
            shear.pivot,
            [0.0, 0.0],
        )

    def test_decompose_2d_shear_recomposition(self):
        print(f'Testing AffineMatrix: decompose 2D shear recomposition ...')
        affine = AffineMatrix.from_array(
            [
                [1.0, 0.5, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        translation, rotation, scale, shear = affine.decompose()

        recomposed = (
            translation
            .compose(rotation)
            .compose(scale)
            .compose(shear)
        )

        np.testing.assert_allclose(
            recomposed._matrix,
            affine._matrix,
            atol=1e-10,
        )

    def test_decompose_3d_shear_recomposition(self):
        print(f'Testing AffineMatrix: decompose 3D shear recomposition ...')
        affine = AffineMatrix.from_array(
            [
                [1.0, 0.2, 0.3, 10.0],
                [0.0, 1.0, 0.4, 20.0],
                [0.0, 0.0, 1.0, 30.0],
            ]
        )

        translation, rotation, scale, shear = affine.decompose()

        recomposed = (
            translation
            .compose(rotation)
            .compose(scale)
            .compose(shear)
        )

        np.testing.assert_allclose(
            recomposed._matrix,
            affine._matrix,
            atol=1e-10,
        )


class TestAffineMatrixPythonProtocols(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_copy(self):
        print(f'Testing AffineMatrix: copy ...')
        affine = AffineMatrix.from_translation(
            [10.0, 20.0],
            pivot=[1.0, 2.0],
        )

        copied = affine.copy()

        self.assertIsNot(copied, affine)
        self.assertEqual(copied, affine)

        copied._matrix[0, 2] = 999.0

        self.assertNotEqual(
            copied.translation[0],
            affine.translation[0],
        )

    def test_matmul(self):
        print(f'Testing AffineMatrix: matmul ...')
        a = AffineMatrix.from_translation([10.0, 20.0])
        b = AffineMatrix.from_translation([1.0, 2.0])

        result = a @ b

        np.testing.assert_allclose(
            result.translation,
            [11.0, 22.0],
        )

    def test_equal(self):
        print(f'Testing AffineMatrix: equal ...')
        a = AffineMatrix.from_translation([10.0, 20.0])
        b = AffineMatrix.from_translation([10.0, 20.0])

        self.assertEqual(a, b)

    def test_not_equal_matrix(self):
        print(f'Testing AffineMatrix: not equal matrix ...')
        a = AffineMatrix.from_translation([10.0, 20.0])
        b = AffineMatrix.from_translation([10.0, 30.0])

        self.assertNotEqual(a, b)

    def test_not_equal_pivot(self):
        print(f'Testing AffineMatrix: not equal pivot ...')
        a = AffineMatrix.identity(2, pivot=[0.0, 0.0])
        b = AffineMatrix.identity(2, pivot=[1.0, 0.0])

        self.assertNotEqual(a, b)

    def test_array(self):
        print(f'Testing AffineMatrix: array ...')
        affine = AffineMatrix.from_translation([10.0, 20.0])

        np.testing.assert_allclose(
            np.asarray(affine),
            affine._matrix,
        )

    def test_array_dtype(self):
        print(f'Testing AffineMatrix: array dtype ...')
        affine = AffineMatrix.identity(2)

        self.assertEqual(
            np.asarray(affine, dtype=np.float32).dtype,
            np.float32,
        )

    def test_repr(self):
        print(f'Testing AffineMatrix: repr ...')
        affine = AffineMatrix.identity(
            2,
            pivot=[10.0, 20.0],
        )

        self.assertEqual(
            repr(affine),
            "AffineMatrix(ndim=2, pivot=[10.0, 20.0])",
        )


class TestAffineMatrixArrayConversion(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_as_homogeneous_2d(self):
        print(f'Testing AffineMatrix: as homogeneous 2D ...')
        affine = AffineMatrix.from_translation([10., 20.])

        expected = np.array([
            [1., 0., 10.],
            [0., 1., 20.],
            [0., 0., 1.]
        ])

        np.testing.assert_allclose(
            affine.as_homogeneous(),
            expected,
        )

    def test_as_compact_2d(self):
        print(f'Testing AffineMatrix: as compact 2D ...')
        affine = AffineMatrix.from_translation([10., 20.])

        expected = np.array([
            [1., 0., 10.],
            [0., 1., 20.]
        ])

        np.testing.assert_allclose(
            affine.as_compact(),
            expected,
        )

    def test_as_homogeneous_3d(self):
        print(f'Testing AffineMatrix: as homogeneous 3D ...')
        affine = AffineMatrix.from_translation([10., 20., 30.])

        expected = np.eye(4)
        expected[:3, 3] = [10., 20., 30.]

        np.testing.assert_allclose(
            affine.as_homogeneous(),
            expected,
        )

    def test_as_compact_3d(self):
        print(f'Testing AffineMatrix: as compact 3D ...')
        affine = AffineMatrix.from_translation([10., 20., 30.])

        expected = np.array([
            [1., 0., 0., 10.],
            [0., 1., 0., 20.],
            [0., 0., 1., 30.]
        ])

        np.testing.assert_allclose(
            affine.as_compact(),
            expected,
        )

    def test_as_array_flatten(self):
        print(f'Testing AffineMatrix: as array flatten ...')
        affine = AffineMatrix.from_translation([10., 20.])

        self.assertEqual(
            affine.as_array(flatten=True).shape,
            (9,),
        )

    def test_as_array_compact_flatten(self):
        print(f'Testing AffineMatrix: as array compact flatten ...')
        affine = AffineMatrix.from_translation([10., 20.])

        self.assertEqual(
            affine.as_array(
                homogeneous=False,
                flatten=True
            ).shape,
            (6,),
        )

    def test_as_array_returns_copy(self):
        print(f'Testing AffineMatrix: as array returns copy ...')
        affine = AffineMatrix.identity(2)

        matrix = affine.as_array()
        matrix[0, 0] = 999.

        self.assertEqual(
            affine.as_homogeneous()[0, 0],
            1.
        )

    def test_as_array_returns_view(self):
        print(f'Testing AffineMatrix: as array returns view ...')
        affine = AffineMatrix.identity(2)

        matrix = affine.as_array(copy=False)
        matrix[0, 0] = 999.

        self.assertEqual(
            affine.as_homogeneous()[0, 0],
            999.
        )


class TestAffineMatrixIO(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_write_and_read_json_2d(self):
        print(f'Testing AffineMatrix: write and read JSON 2D ...')
        import tempfile
        from pathlib import Path

        affine = AffineMatrix.from_array(
            [
                [1.0, 0.2, 10.0],
                [0.3, 1.0, 20.0],
            ],
            pivot=[100.0, 200.0],
        )

        with tempfile.TemporaryDirectory() as directory:
            filepath = Path(directory) / "affine.json"

            affine.write(filepath)
            loaded = AffineMatrix.read(filepath)

        self.assertEqual(loaded, affine)

    def test_write_and_read_json_3d(self):
        print(f'Testing AffineMatrix: write and read JSON 3D ...')
        import tempfile
        from pathlib import Path

        affine = AffineMatrix.from_array(
            [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 20.0],
                [0.0, 0.0, 1.0, 30.0],
            ],
            pivot=[100.0, 200.0, 300.0],
        )

        with tempfile.TemporaryDirectory() as directory:
            filepath = Path(directory) / "affine.json"

            affine.write(filepath)
            loaded = AffineMatrix.read(filepath)

        self.assertEqual(loaded, affine)

    def test_write_json_structure(self):
        print(f'Testing AffineMatrix: write JSON structure ...')
        import json
        import tempfile
        from pathlib import Path

        affine = AffineMatrix.from_translation(
            [10.0, 20.0],
            pivot=[1.0, 2.0],
        )

        with tempfile.TemporaryDirectory() as directory:
            filepath = Path(directory) / "affine.json"
            affine.write(filepath)

            with filepath.open("r", encoding="utf-8") as file:
                data = json.load(file)

        self.assertEqual(
            data["transform"],
            [1.0, 0.0, 10.0, 0.0, 1.0, 20.0],
        )
        self.assertEqual(
            data["pivot"],
            [1.0, 2.0],
        )

    def test_write_and_read_csv_2d(self):
        print(f'Testing AffineMatrix: write and read CSV 2D ...')
        import tempfile
        from pathlib import Path

        affine = AffineMatrix.from_array(
            [
                [1.0, 0.2, 10.0],
                [0.3, 1.0, 20.0],
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            filepath = Path(directory) / "affine.csv"

            affine.write(filepath)
            loaded = AffineMatrix.read(filepath)

        self.assertEqual(loaded, affine)

    def test_write_and_read_csv_3d(self):
        print(f'Testing AffineMatrix: write and read CSV 3D ...')
        import tempfile
        from pathlib import Path

        affine = AffineMatrix.from_array(
            [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 20.0],
                [0.0, 0.0, 1.0, 30.0],
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            filepath = Path(directory) / "affine.csv"

            affine.write(filepath)
            loaded = AffineMatrix.read(filepath)

        self.assertEqual(loaded, affine)

    def test_csv_does_not_preserve_pivot(self):
        print(f'Testing AffineMatrix: CSV does not preserve pivot ...')
        import tempfile
        from pathlib import Path

        affine = AffineMatrix.identity(
            2,
            pivot=[10.0, 20.0],
        )

        with tempfile.TemporaryDirectory() as directory:
            filepath = Path(directory) / "affine.csv"

            affine.write(filepath)
            loaded = AffineMatrix.read(filepath)

        np.testing.assert_allclose(
            loaded.pivot,
            [0.0, 0.0],
        )

    def test_write_rejects_unsupported_file_type(self):
        print(f'Testing AffineMatrix: write rejects unsupported file type ...')
        import tempfile
        from pathlib import Path

        affine = AffineMatrix.identity(2)

        with tempfile.TemporaryDirectory() as directory:
            filepath = Path(directory) / "affine.txt"

            with self.assertRaisesRegex(
                ValueError,
                "Unsupported file type",
            ):
                affine.write(filepath)

    def test_read_rejects_unsupported_file_type(self):
        print(f'Testing AffineMatrix: read rejects unsupported file type ...')
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            filepath = Path(directory) / "affine.txt"
            filepath.write_text("invalid")

            with self.assertRaisesRegex(
                ValueError,
                "Unsupported file type",
            ):
                AffineMatrix.read(filepath)

    def test_read_json_rejects_missing_transform(self):
        print(f'Testing AffineMatrix: read JSON rejects missing transform ...')
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            filepath = Path(directory) / "affine.json"

            with filepath.open("w", encoding="utf-8") as file:
                json.dump({"pivot": [0.0, 0.0]}, file)

            with self.assertRaisesRegex(
                ValueError,
                "must contain a 'transform'",
            ):
                AffineMatrix.read(filepath)


class TestAffineMatrixElastix(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_to_elastix_round_trip_2d(self):
        print(f'Testing AffineMatrix: to elastix round trip 2D ...')
        from squirrel.library.elastix import elastix_to_c

        affine = AffineMatrix.from_array(
            [
                [1.1, 0.2, 10.0],
                [0.3, 0.9, 20.0],
            ]
        )

        elastix_parameters = affine.to_elastix()

        compact = elastix_to_c(
            "AffineTransform",
            elastix_parameters,
        )

        reconstructed = AffineMatrix.from_array(compact)

        np.testing.assert_allclose(
            reconstructed.as_compact(),
            affine.as_compact(),
            atol=1e-12,
        )

    def test_to_elastix_parameter_map(self):
        print(f'Testing AffineMatrix: to elastix parameter map ...')
        try:
            import SimpleITK
        except ImportError:
            self.skipTest("SimpleITK is not installed.")

        affine = AffineMatrix.identity(
            2,
            pivot=[10.0, 20.0],
        )

        parameter_map = affine.to_elastix(
            shape=[100, 200],
            as_parameter_map=True,
        )

        self.assertEqual(
            parameter_map["Transform"],
            ("AffineTransform",),
        )
        self.assertEqual(
            parameter_map["CenterOfRotationPoint"],
            ("20.0", "10.0"),
        )
        self.assertEqual(
            parameter_map["Size"],
            ("200", "100"),
        )


if __name__ == "__main__":
    unittest.main()



# class TestAffineMatrix(unittest.TestCase):

#     def setUp(self):
#         warnings.simplefilter('ignore', category=Warning)

#     def test_inverse(self):
#         print(f'Testing AffineMatrix: inverse ...')
#         params = np.array([
#             [0.3, -0.02, 1],
#             [0.12, 1.2, 2]
#         ])
#         A = AffineMatrix(parameters=params)
#         B = A.inverse()

#         I = (A * B).get_matrix('Ms')
#         self.assertTrue(np.allclose(I, np.eye(3), atol=1e-6))

#     def test_dot(self):
#         print(f'Testing AffineMatrix: dot ...')
#         params_a = np.array([
#             [1, 0, 0],
#             [0.12, 1.2, 2]
#         ])
#         params_b = np.array([
#             [0.3, -0.02, 1],
#             [0, 1, 0]
#         ])
#         A = AffineMatrix(parameters=params_a)
#         B = AffineMatrix(parameters=params_b)
#         C = A * B
#         t = C.get_matrix('M')
#         out = np.array([
#             [0.3, -0.02, 1],
#             [0.036, 1.1976, 2.12]
#         ])
#         self.assertTrue(np.allclose(t, out))

#     def test_get_translation(self):
#         print(f'Testing AffineMatrix: get_translation ...')
#         t = [3, -2]
#         A = AffineMatrix(translation=t)
#         self.assertTrue(np.allclose(A.get_translation(), t))

#     def test_set_translation(self):
#         print(f'Testing AffineMatrix: set_translation ...')
#         A = AffineMatrix(translation=[0, 0])
#         A.set_translation([5, -3])
#         self.assertTrue(np.allclose(A.get_translation(), [5, -3]))

#     def test_get_scaled(self):
#         print(f'Testing AffineMatrix: get_scaled ...')
#         params = np.array([
#             [0.3, -0.02, 1],
#             [0.12, 1.2, 2]
#         ])
#         A = AffineMatrix(parameters=params, pivot=[10, 10])
#         B = A.get_scaled(0.5)

#         out = np.array([
#             [0.3, -0.02, -1.7],
#             [0.12,  1.2,  9.2]
#         ])

#         self.assertTrue(np.allclose(B.get_matrix('M'), out))

#     def test_decompose(self):
#         # translation + scale (no rotation to keep it simple)
#         T = AffineMatrix(translation=[2, 3])
#         S = AffineMatrix(parameters=[[2, 0, 0],
#                                      [0, 3, 0]])

#         A = T * S

#         t, r, z, s = A.decompose()

#         self.assertTrue(np.allclose(t.get_translation(), [2, 3]))
#         self.assertTrue(np.allclose(z.get_matrix('M')[:2, :2], [[2, 0], [0, 3]], atol=1e-6))

#     def test_shift_pivot_to_origin(self):
#         A = AffineMatrix(translation=[1, 0], pivot=[2, 0])
#         A.shift_pivot_to_origin()

#         self.assertTrue(np.allclose(A.get_pivot(), [0, 0]))

#     def test_return_3d(self):
#         A = AffineMatrix(translation=[1, 2], pivot=[3, 4])
#         B = A.return_3d(axis=2)

#         self.assertEqual(B.get_ndim(), 3)

#         t = B.get_translation()
#         self.assertTrue(np.allclose(t[:2], [1, 2]))
#         self.assertTrue(np.allclose(t[2], 0))

#         pivot = B.get_pivot()
#         self.assertTrue(np.allclose(pivot, [3, 4, 0]))
