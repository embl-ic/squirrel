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


class TestAffineWorkflowCompatibility(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_sequence_and_crop_preserve_metadata(self):
        print('Testing affine workflow sequencing and cropping')
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from squirrel.library.affine_matrices import AffineStack
        from squirrel.workflows.transformation import (
            sequence_affine_stack_workflow,
            crop_transform_sequence_workflow,
        )

        with TemporaryDirectory() as folder:
            source = Path(folder) / 'source.json'
            sequenced = Path(folder) / 'sequenced.json'
            cropped = Path(folder) / 'cropped.json'
            stack = AffineStack.from_array(
                [[1, 0, 1, 0, 1, 0]] * 4,
                sequenced=False,
                metadata={'stack_shape': [4, 20, 30]},
            )
            stack.write(source)
            sequence_affine_stack_workflow(source, sequenced)
            result = AffineStack.read(sequenced)
            self.assertTrue(result.sequenced)
            np.testing.assert_allclose([m.translation[0] for m in result], [1, 2, 3, 4])

            crop_transform_sequence_workflow(sequenced, cropped, [1, 3])
            result = AffineStack.read(cropped)
            self.assertEqual(len(result), 2)
            self.assertTrue(result.sequenced)
            self.assertEqual(result.get_metadata('stack_shape'), [2, 20, 30])
            np.testing.assert_allclose([m.translation[0] for m in result], [2, 3])

    def test_drift_and_smoothing(self):
        print('Testing affine workflow drift and smoothing')
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from squirrel.library.affine_matrices import AffineStack
        from squirrel.workflows.transformation import (
            add_translational_drift_workflow,
            smooth_affine_sequence_workflow,
        )

        with TemporaryDirectory() as folder:
            source = Path(folder) / 'source.json'
            drifted = Path(folder) / 'drifted.json'
            smoothed = Path(folder) / 'smoothed.json'
            stack = AffineStack.from_array([[1, 0, 0, 0, 1, 0]] * 3, sequenced=True)
            stack.write(source)
            add_translational_drift_workflow(source, drifted, [2, 3])
            result = AffineStack.read(drifted)
            np.testing.assert_allclose([m.translation for m in result], [[0, 0], [2, 3], [4, 6]])
            smooth_affine_sequence_workflow(drifted, smoothed, sigma=0)
            np.testing.assert_allclose(AffineStack.read(smoothed).as_array(), result.as_array())


class TestModifyStepInSequenceWorkflow(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_modify_step_in_sequence_workflow_relative_and_replace(self):
        print('Testing modify_step_in_sequence_workflow: relative composition and replacement')

        from squirrel.library.affine_matrices import AffineStack
        from squirrel.workflows.transformation import modify_step_in_sequence_workflow

        stack = AffineStack.from_array(
            [[1, 0, 1, 0, 1, 2], [1, 0, 3, 0, 1, 4]],
            pivot=[5, 7], metadata={'source': 'test'},
        )
        affine = [1, 0, 10, 0, 1, 20]
        result = modify_step_in_sequence_workflow(stack, None, 1, affine, return_result=True)
        np.testing.assert_allclose(result[0].translation, [1, 2])
        np.testing.assert_allclose(result[1].translation, [13, 24])
        np.testing.assert_allclose(stack[1].translation, [3, 4])
        self.assertEqual(result.metadata, stack.metadata)
        np.testing.assert_allclose(result.pivot, [5, 7])
        self.assertFalse(result.sequenced)

        replaced = modify_step_in_sequence_workflow(
            stack, None, 1, affine, replace=True, return_result=True,
        )
        np.testing.assert_allclose(replaced[1].translation, [10, 20])

    def test_modify_step_in_sequence_workflow_sequenced_suffix(self):
        print('Testing modify_step_in_sequence_workflow: sequenced suffix and validation')

        from squirrel.library.affine_matrices import AffineStack
        from squirrel.workflows.transformation import modify_step_in_sequence_workflow

        stack = AffineStack.from_array(
            [[1, 0, 1, 0, 1, 2], [1, 0, 3, 0, 1, 4], [1, 0, 5, 0, 1, 6]],
            sequenced=True,
        )
        result = modify_step_in_sequence_workflow(
            stack, None, 1, [1, 0, 10, 0, 1, 20], return_result=True,
        )
        np.testing.assert_allclose(
            [matrix.translation for matrix in result],
            [[1, 2], [13, 24], [15, 26]],
        )
        self.assertTrue(result.sequenced)
        with self.assertRaises(ValueError):
            modify_step_in_sequence_workflow(
                stack, None, 1, [1, 0, 10, 0, 1, 20], replace=True,
            )
        with self.assertRaises(IndexError):
            modify_step_in_sequence_workflow(
                stack, None, 3, [1, 0, 10, 0, 1, 20], return_result=True,
            )
