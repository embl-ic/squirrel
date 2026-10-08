import unittest
import warnings
import os
from shutil import rmtree
import numpy as np

from random import randint

from squirrel.library.transformation import apply_affine_transform
from squirrel.library.affine_matrices import AffineMatrix


class TestApplyAffineTransform(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_identity_2d(self):
        print(f'Testing transformation: apply affine identity 2D ...')

        image = np.arange(25, dtype=np.float32).reshape(5, 5)
        transform = AffineMatrix.identity(2)

        result = apply_affine_transform(
            image,
            transform,
            no_offset_to_center=True,
            order=0,
        )

        np.testing.assert_array_equal(
            result,
            image,
        )

    def test_translation_2d(self):
        print(f'Testing transformation: apply affine translation 2D ...')

        image = np.zeros((7, 7), dtype=np.float32)
        image[3, 3] = 1.

        transform = AffineMatrix.from_translation([1., 2.])

        result = apply_affine_transform(
            image,
            transform,
            no_offset_to_center=True,
            fill_mode='constant',
            cval=0.,
            order=0,
        )

        expected = np.zeros_like(image)
        expected[4, 5] = 1.

        np.testing.assert_array_equal(
            result,
            expected,
        )

    def test_translation_from_array(self):
        print(f'Testing transformation: apply affine from array ...')

        image = np.zeros((7, 7), dtype=np.float32)
        image[2, 2] = 1.

        transform = np.array([
            [1., 0., 2.],
            [0., 1., 1.],
        ])

        result = apply_affine_transform(
            image,
            transform,
            no_offset_to_center=True,
            fill_mode='constant',
            cval=0.,
            order=0,
        )

        expected = np.zeros_like(image)
        expected[4, 3] = 1.

        np.testing.assert_array_equal(
            result,
            expected,
        )

    def test_explicit_pivot(self):
        print(f'Testing transformation: apply affine with explicit pivot ...')

        image = np.zeros((7, 7), dtype=np.float32)
        image[3, 4] = 1.

        transform = AffineMatrix.from_rotation(
            np.pi / 2,
        )

        result = apply_affine_transform(
            image,
            transform,
            pivot=[3., 3.],
            no_offset_to_center=True,
            fill_mode='constant',
            cval=0.,
            order=0,
        )

        expected = np.zeros_like(image)
        expected[2, 3] = 1.

        np.testing.assert_array_equal(
            result,
            expected,
        )

    def test_defaults_to_image_center_pivot(self):
        print(f'Testing transformation: apply affine with image center pivot ...')

        image = np.zeros((6, 6), dtype=np.float32)
        image[3, 4] = 1.

        transform = AffineMatrix.from_rotation(
            np.pi / 2,
        )

        result = apply_affine_transform(
            image,
            transform,
            fill_mode='constant',
            cval=0.,
            order=0,
        )

        expected = np.zeros_like(image)
        expected[2, 3] = 1.

        np.testing.assert_array_equal(
            result,
            expected,
        )

    def test_no_offset_to_center_preserves_transform_pivot(self):
        print(f'Testing transformation: no offset to center ...')

        image = np.zeros((7, 7), dtype=np.float32)
        image[1, 0] = 1.

        transform = AffineMatrix.from_rotation(
            np.pi / 2,
            pivot=[0., 0.],
        )

        result = apply_affine_transform(
            image,
            transform,
            no_offset_to_center=True,
            fill_mode='constant',
            cval=0.,
            order=0,
        )

        expected = np.zeros_like(image)
        expected[0, 1] = 1.

        np.testing.assert_array_equal(
            result,
            expected,
        )

    def test_does_not_modify_transform(self):
        print(f'Testing transformation: apply affine does not modify transform ...')

        image = np.zeros((5, 5), dtype=np.float32)

        transform = AffineMatrix.from_rotation(
            np.pi / 4,
            pivot=[0., 0.],
        )
        original = transform.copy()

        apply_affine_transform(
            image,
            transform,
            pivot=[2., 2.],
            order=0,
        )

        self.assertEqual(
            transform,
            original,
        )

    def test_rejects_dimension_mismatch(self):
        print(f'Testing transformation: apply affine rejects dimension mismatch ...')

        image = np.zeros((5, 5), dtype=np.float32)
        transform = AffineMatrix.identity(3)

        with self.assertRaisesRegex(
            ValueError,
            "Transform is 3D",
        ):
            apply_affine_transform(
                image,
                transform,
            )

    def test_rejects_invalid_apply_mode(self):
        print(f'Testing transformation: apply affine rejects invalid mode ...')

        image = np.zeros((5, 5), dtype=np.float32)
        transform = AffineMatrix.identity(2)

        with self.assertRaisesRegex(
            ValueError,
            "Invalid apply mode",
        ):
            apply_affine_transform(
                image,
                transform,
                apply='invalid',
            )

    def test_rotation_mode_not_implemented(self):
        print(f'Testing transformation: rotation-only mode not implemented ...')

        image = np.zeros((5, 5), dtype=np.float32)
        transform = AffineMatrix.identity(2)

        with self.assertRaises(NotImplementedError):
            apply_affine_transform(
                image,
                transform,
                apply='rotation',
            )

    def test_scale_canvas_requires_no_offset_to_center(self):
        print(f'Testing transformation: scale canvas requires no offset to center ...')

        image = np.zeros((10, 10), dtype=np.float32)
        transform = AffineMatrix.from_scale([2., 2.])

        with self.assertRaisesRegex(
            ValueError,
            "requires no_offset_to_center=True",
        ):
            apply_affine_transform(
                image,
                transform,
                scale_canvas=True,
            )

    def test_scale_canvas_requires_zero_pivot(self):
        print(f'Testing transformation: scale canvas requires zero pivot ...')

        image = np.zeros((10, 10), dtype=np.float32)

        transform = AffineMatrix.from_scale(
            [2., 2.],
            pivot=[5., 5.],
        )

        with self.assertRaisesRegex(
            ValueError,
            "requires a zero pivot",
        ):
            apply_affine_transform(
                image,
                transform,
                no_offset_to_center=True,
                scale_canvas=True,
            )

    def test_scale_canvas_crops_result(self):
        print(f'Testing transformation: scale canvas crops result ...')

        image = np.zeros((10, 12), dtype=np.float32)
        transform = AffineMatrix.from_scale([2., 3.])

        result = apply_affine_transform(
            image,
            transform,
            no_offset_to_center=True,
            scale_canvas=True,
            order=0,
        )

        self.assertEqual(
            result.shape,
            (5, 4),
        )
