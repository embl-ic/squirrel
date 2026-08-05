
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