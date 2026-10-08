
import tempfile
import unittest
from pathlib import Path
import warnings

import numpy as np

from squirrel.library.affine_matrices import AffineMatrix
from squirrel.workflows.transformation import (
    decompose_affine_workflow, 
    apply_affine_workflow, 
    apply_affines_workflow
)


class TestDecomposeAffineWorkflow(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)


    def test_from_affine_matrix(self):
        transform = (
            AffineMatrix.from_translation([10., 20.])
            @ AffineMatrix.from_rotation(np.pi / 4)
            @ AffineMatrix.from_scale([2., 3.])
        )

        decomposition = decompose_affine_workflow(
            transform
        )

        recomposed = decomposition[0]

        for component in decomposition[1:]:
            recomposed = recomposed @ component

        np.testing.assert_allclose(
            recomposed.as_homogeneous(),
            transform
            .shifted_pivot_to_origin()
            .as_homogeneous(),
            atol=1e-10,
        )
    def test_from_array(self):
        transform = np.array([
            [2., 0., 10.],
            [0., 3., 20.],
        ])

        translation, rotation, scale, shear = (
            decompose_affine_workflow(transform)
        )

        np.testing.assert_allclose(
            translation.translation,
            [10., 20.],
            atol=1e-12,
        )

        np.testing.assert_allclose(
            rotation.linear,
            np.eye(2),
            atol=1e-12,
        )

        np.testing.assert_allclose(
            scale.linear,
            np.diag([2., 3.]),
            atol=1e-12,
        )

        np.testing.assert_allclose(
            shear.linear,
            np.eye(2),
            atol=1e-12,
        )
    def test_accepts_input_pivot(self):
        transform = AffineMatrix.from_rotation(
            np.pi / 2
        )

        decomposition = decompose_affine_workflow(
            transform,
            pivot=[10., 20.],
        )

        recomposed = decomposition[0]

        for component in decomposition[1:]:
            recomposed = recomposed @ component

        expected = (
            transform
            .with_pivot([10., 20.])
            .shifted_pivot_to_origin()
        )

        np.testing.assert_allclose(
            recomposed.as_homogeneous(),
            expected.as_homogeneous(),
            atol=1e-10,
        )
    def test_uses_shear_to_translation_pivot(self):
        transform = AffineMatrix.from_array([
            [1., 0.5, 0.],
            [0., 1., 0.],
        ])

        translation, rotation, scale, shear = (
            decompose_affine_workflow(
                transform,
                shear_to_translation_pivot=[10., 20.],
            )
        )

        self.assertFalse(
            np.allclose(
                translation.translation,
                [0., 0.],
            )
        )
    def test_reads_transform_from_file(self):
        transform = AffineMatrix.from_translation(
            [10., 20.]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "transform.json"
            transform.write(filepath)

            translation, rotation, scale, shear = (
                decompose_affine_workflow(filepath)
            )

        np.testing.assert_allclose(
            translation.translation,
            [10., 20.],
        )
    def test_writes_decomposition(self):
        transform = (
            AffineMatrix.from_translation([10., 20.])
            @ AffineMatrix.from_scale([2., 3.])
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_folder = Path(tmpdir) / "decomposition"

            returned = decompose_affine_workflow(
                transform,
                out_folder=out_folder,
            )

            loaded = tuple(
                AffineMatrix.read(
                    out_folder / f"{name}.json"
                )
                for name in (
                    "translation",
                    "rotation",
                    "scale",
                    "shear",
                )
            )

            for name in (
                "translation",
                "rotation",
                "scale",
                "shear",
            ):
                self.assertTrue(
                    (
                        out_folder / f"{name}.json"
                    ).is_file()
                )

        for returned_component, loaded_component in zip(
            returned,
            loaded,
        ):
            self.assertEqual(
                returned_component,
                loaded_component,
            )
    def test_does_not_modify_input_instance(self):
        transform = AffineMatrix.from_rotation(
            np.pi / 4
        )

        original = transform.copy()

        decompose_affine_workflow(
            transform,
            pivot=[10., 20.],
        )

        self.assertEqual(
            transform,
            original,
        )


class TestApplyAffineWorkflow(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_from_image_array_and_affine_matrix(self):
        print(f'Testing workflow: apply affine from array and AffineMatrix ...')

        image = np.zeros((7, 7), dtype=np.float32)
        image[3, 3] = 1.

        transform = AffineMatrix.from_translation([1., 2.])

        result = apply_affine_workflow(
            image,
            transform,
            no_offset_to_center=True,
        )

        self.assertEqual(
            result.shape,
            image.shape,
        )

        self.assertGreater(
            result[4, 5],
            0.,
        )

    def test_from_transform_array(self):
        print(f'Testing workflow: apply affine from transform array ...')

        image = np.zeros((7, 7), dtype=np.float32)
        image[2, 2] = 1.

        transform = np.array([
            [1., 0., 2.],
            [0., 1., 1.],
        ])

        result = apply_affine_workflow(
            image,
            transform,
            no_offset_to_center=True,
        )

        self.assertEqual(
            result.shape,
            image.shape,
        )

    def test_from_transform_file(self):
        print(f'Testing workflow: apply affine from transform file ...')

        image = np.zeros((7, 7), dtype=np.float32)
        image[3, 3] = 1.

        transform = AffineMatrix.from_translation([1., 2.])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'transform.json'
            transform.write(filepath)

            result = apply_affine_workflow(
                image,
                filepath,
                no_offset_to_center=True,
            )

        self.assertEqual(
            result.shape,
            image.shape,
        )

        self.assertGreater(
            result[4, 5],
            0.,
        )

    def test_explicit_pivot(self):
        print(f'Testing workflow: apply affine with explicit pivot ...')

        image = np.zeros((7, 7), dtype=np.float32)
        image[3, 4] = 1.

        transform = AffineMatrix.from_rotation(
            np.pi / 2,
        )

        result = apply_affine_workflow(
            image,
            transform,
            pivot=[3., 3.],
            no_offset_to_center=True,
        )

        self.assertGreater(
            result[4, 3],
            0.,
        )

    def test_does_not_modify_transform(self):
        print(f'Testing workflow: apply affine does not modify transform ...')

        image = np.zeros((7, 7), dtype=np.float32)

        transform = AffineMatrix.from_rotation(
            np.pi / 4,
            pivot=[0., 0.],
        )
        original = transform.copy()

        apply_affine_workflow(
            image,
            transform,
            pivot=[3., 3.],
        )

        self.assertEqual(
            transform,
            original,
        )

    def test_without_output_file(self):
        print(f'Testing workflow: apply affine without output file ...')

        image = np.arange(
            25,
            dtype=np.float32,
        ).reshape(5, 5)

        transform = AffineMatrix.identity(2)

        result = apply_affine_workflow(
            image,
            transform,
            out_filepath=None,
            no_offset_to_center=True,
        )

        np.testing.assert_allclose(
            result,
            image,
            atol=1e-12
        )

    def test_writes_output_file(self):
        print(f'Testing workflow: apply affine writes output file ...')

        import h5py

        image = np.arange(
            25,
            dtype=np.float32,
        ).reshape(5, 5)

        transform = AffineMatrix.identity(2)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'result.h5'

            returned = apply_affine_workflow(
                image,
                transform,
                out_filepath=filepath,
                no_offset_to_center=True,
            )

            self.assertTrue(
                filepath.is_file()
            )

            with h5py.File(filepath, 'r') as handle:
                written = handle['data'][:]

        np.testing.assert_allclose(
            written,
            returned,
        )

    def test_reads_image_file(self):
        print(f'Testing workflow: apply affine reads image file ...')

        import h5py

        image = np.arange(
            25,
            dtype=np.float32,
        ).reshape(5, 5)

        transform = AffineMatrix.identity(2)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'image.h5'

            with h5py.File(filepath, 'w') as handle:
                handle.create_dataset(
                    'input',
                    data=image,
                )

            result = apply_affine_workflow(
                filepath,
                transform,
                image_key='input',
                no_offset_to_center=True,
            )

        np.testing.assert_allclose(
            result,
            image,
            atol=1e-12
        )

    def test_rejects_dimension_mismatch(self):
        print(f'Testing workflow: apply affine rejects dimension mismatch ...')

        image = np.zeros((5, 5), dtype=np.float32)
        transform = AffineMatrix.identity(3)

        with self.assertRaisesRegex(
            ValueError,
            "Transform is 3D",
        ):
            apply_affine_workflow(
                image,
                transform,
            )

    def test_scale_canvas(self):
        print(f'Testing workflow: apply affine scale canvas ...')

        image = np.zeros((10, 12), dtype=np.float32)
        transform = AffineMatrix.from_scale([2., 3.])

        result = apply_affine_workflow(
            image,
            transform,
            no_offset_to_center=True,
            scale_canvas=True,
        )

        self.assertEqual(
            result.shape,
            (5, 4),
        )


class TestApplyAffinesWorkflow(unittest.TestCase):

    def test_single_transform(self):
        print(f'Testing workflow: apply multiple affines with one transform ...')

        image = np.zeros((7, 7), dtype=np.float32)
        image[3, 3] = 1.

        transform = AffineMatrix.from_translation([1., 2.])

        expected = apply_affine_workflow(
            image,
            transform,
            no_offset_to_center=True,
        )

        result = apply_affines_workflow(
            image,
            [transform],
            no_offset_to_center=True,
        )

        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-12,
        )

    def test_two_translations(self):
        print(f'Testing workflow: apply multiple affine translations ...')

        image = np.zeros((9, 9), dtype=np.float32)
        image[3, 3] = 1.

        first = AffineMatrix.from_translation([1., 2.])
        second = AffineMatrix.from_translation([2., 1.])

        expected_transform = first @ second

        expected = apply_affine_workflow(
            image,
            expected_transform,
            no_offset_to_center=True,
        )

        result = apply_affines_workflow(
            image,
            [first, second],
            no_offset_to_center=True,
        )

        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-12,
        )

    def test_affine_matrix_inputs(self):
        print(f'Testing workflow: apply multiple affines from AffineMatrix objects ...')

        image = np.zeros((9, 9), dtype=np.float32)
        image[4, 4] = 1.

        transforms = [
            AffineMatrix.from_translation([1., 0.]),
            AffineMatrix.from_translation([0., 2.]),
        ]

        result = apply_affines_workflow(
            image,
            transforms,
            no_offset_to_center=True,
        )

        expected = apply_affine_workflow(
            image,
            transforms[0] @ transforms[1],
            no_offset_to_center=True,
        )

        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-12,
        )

    def test_array_inputs(self):
        print(f'Testing workflow: apply multiple affines from arrays ...')

        image = np.zeros((9, 9), dtype=np.float32)
        image[4, 4] = 1.

        first = np.array([
            [1., 0., 1.],
            [0., 1., 0.],
        ])

        second = np.array([
            [1., 0., 0.],
            [0., 1., 2.],
        ])

        result = apply_affines_workflow(
            image,
            [first, second],
            no_offset_to_center=True,
        )

        expected_transform = (
            AffineMatrix.from_array(first)
            @ AffineMatrix.from_array(second)
        )

        expected = apply_affine_workflow(
            image,
            expected_transform,
            no_offset_to_center=True,
        )

        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-12,
        )

    def test_transform_file_inputs(self):
        print(f'Testing workflow: apply multiple affines from files ...')

        image = np.zeros((9, 9), dtype=np.float32)
        image[4, 4] = 1.

        first = AffineMatrix.from_translation([1., 0.])
        second = AffineMatrix.from_translation([0., 2.])

        with tempfile.TemporaryDirectory() as tmpdir:
            first_path = Path(tmpdir) / 'first.json'
            second_path = Path(tmpdir) / 'second.json'

            first.write(first_path)
            second.write(second_path)

            result = apply_affines_workflow(
                image,
                [first_path, second_path],
                no_offset_to_center=True,
            )

        expected = apply_affine_workflow(
            image,
            first @ second,
            no_offset_to_center=True,
        )

        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-12,
        )

    def test_mixed_transform_inputs(self):
        print(f'Testing workflow: apply multiple affines from mixed inputs ...')

        image = np.zeros((9, 9), dtype=np.float32)
        image[4, 4] = 1.

        first = AffineMatrix.from_translation([1., 0.])
        second = np.array([
            [1., 0., 0.],
            [0., 1., 2.],
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            first_path = Path(tmpdir) / 'first.json'
            first.write(first_path)

            result = apply_affines_workflow(
                image,
                [first_path, second],
                no_offset_to_center=True,
            )

        expected_transform = (
            first
            @ AffineMatrix.from_array(second)
        )

        expected = apply_affine_workflow(
            image,
            expected_transform,
            no_offset_to_center=True,
        )

        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-12,
        )

    def test_empty_transform_list(self):
        print(f'Testing workflow: apply multiple affines rejects empty list ...')

        image = np.zeros((5, 5), dtype=np.float32)

        with self.assertRaisesRegex(
            ValueError,
            "at least one affine",
        ):
            apply_affines_workflow(
                image,
                [],
            )

    def test_explicit_pivot(self):
        print(f'Testing workflow: apply multiple affines with explicit pivot ...')

        image = np.zeros((7, 7), dtype=np.float32)
        image[3, 4] = 1.

        transform = AffineMatrix.from_rotation(
            np.pi / 2,
        )

        result = apply_affines_workflow(
            image,
            [transform],
            pivot=[3., 3.],
            no_offset_to_center=True,
        )

        expected = apply_affine_workflow(
            image,
            transform.with_pivot([3., 3.]),
            pivot=[3., 3.],
            no_offset_to_center=True,
        )

        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-12,
        )

    def test_does_not_modify_input_transforms(self):
        print(f'Testing workflow: apply multiple affines does not modify inputs ...')

        image = np.zeros((7, 7), dtype=np.float32)

        first = AffineMatrix.from_translation([1., 2.])
        second = AffineMatrix.from_rotation(np.pi / 4)

        first_original = first.copy()
        second_original = second.copy()

        apply_affines_workflow(
            image,
            [first, second],
            pivot=[3., 3.],
        )

        self.assertEqual(
            first,
            first_original,
        )
        self.assertEqual(
            second,
            second_original,
        )

    def test_without_output_file(self):
        print(f'Testing workflow: apply multiple affines without output file ...')

        image = np.arange(
            25,
            dtype=np.float32,
        ).reshape(5, 5)

        result = apply_affines_workflow(
            image,
            [AffineMatrix.identity(2)],
            out_filepath=None,
            no_offset_to_center=True,
        )

        np.testing.assert_allclose(
            result,
            image,
            atol=1e-12,
        )

    def test_writes_composed_transform(self):
        print(f'Testing workflow: apply multiple affines writes composed transform ...')

        image = np.zeros((7, 7), dtype=np.float32)

        first = AffineMatrix.from_translation([1., 2.])
        second = AffineMatrix.from_translation([3., 4.])

        expected_transform = first @ second

        with tempfile.TemporaryDirectory() as tmpdir:
            out_filepath = Path(tmpdir) / 'result.h5'
            transform_filepath = out_filepath.with_suffix('.json')

            apply_affines_workflow(
                image,
                [first, second],
                out_filepath=out_filepath,
                no_offset_to_center=True,
            )

            self.assertTrue(
                transform_filepath.is_file()
            )

            written_transform = AffineMatrix.read(
                transform_filepath
            )

        self.assertEqual(
            written_transform,
            expected_transform,
        )

    def test_writes_transformed_image(self):
        print(f'Testing workflow: apply multiple affines writes transformed image ...')

        import h5py

        image = np.arange(
            25,
            dtype=np.float32,
        ).reshape(5, 5)

        transforms = [
            AffineMatrix.identity(2),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_filepath = Path(tmpdir) / 'result.h5'

            returned = apply_affines_workflow(
                image,
                transforms,
                out_filepath=out_filepath,
                no_offset_to_center=True,
            )

            self.assertTrue(
                out_filepath.is_file()
            )

            with h5py.File(out_filepath, 'r') as handle:
                written = handle['data'][:]

        np.testing.assert_allclose(
            written,
            returned,
            atol=1e-12,
        )

    def test_composition_order(self):
        print(f'Testing workflow: apply multiple affines composition order ...')

        image = np.zeros((9, 9), dtype=np.float32)
        image[3, 4] = 1.

        rotation = AffineMatrix.from_rotation(
            np.pi / 2,
            pivot=[3., 3.],
        )
        translation = AffineMatrix.from_translation(
            [1., 0.],
            pivot=[3., 3.],
        )

        result = apply_affines_workflow(
            image,
            [rotation, translation],
            pivot=[3., 3.],
            no_offset_to_center=True,
        )

        expected = apply_affine_workflow(
            image,
            rotation @ translation,
            pivot=[3., 3.],
            no_offset_to_center=True,
        )

        reverse = apply_affine_workflow(
            image,
            translation @ rotation,
            pivot=[3., 3.],
            no_offset_to_center=True,
        )

        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-12,
        )

        self.assertFalse(
            np.allclose(
                result,
                reverse,
                atol=1e-12,
            )
        )
