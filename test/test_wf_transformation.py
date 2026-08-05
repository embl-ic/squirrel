
import tempfile
import unittest
from pathlib import Path
import warnings

import numpy as np

from squirrel.library.affine_matrices import AffineMatrix
from squirrel.workflows.transformation import decompose_affine_workflow


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


from squirrel.workflows.transformation import apply_affine_workflow


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
