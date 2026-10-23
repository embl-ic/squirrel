import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

from squirrel.library.affine_matrices import AffineMatrix, AffineStack
from squirrel.linear_algebra import apply_z_step, append_affine_stack


class TestLinearAlgebraAffineCompatibility(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_apply_z_step_converts_relative_stack(self):
        print('Testing apply_z_step: relative input and metadata ...')

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.json'
            target = Path(directory) / 'output.json'
            stack = AffineStack(
                matrices=[AffineMatrix.from_translation([float(idx), 0.]) for idx in range(3)],
                sequenced=False,
                metadata={'z_step': 1},
            )
            stack.write(source)

            with patch.object(sys, 'argv', ['apply_z_step', str(source), str(target)]):
                apply_z_step()

            result = AffineStack.read(target)
            self.assertTrue(result.sequenced)
            self.assertEqual(result.get_metadata('z_step'), 1)
            np.testing.assert_allclose(result[2].translation, [3., 0.])

    def test_append_affine_stack_preserves_order_and_metadata(self):
        print('Testing append_affine_stack: append all matrices ...')

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / 'first.json'
            second = Path(directory) / 'second.json'
            target = Path(directory) / 'output.json'
            AffineStack(
                matrices=[AffineMatrix.from_translation([1., 0.])],
                metadata={'labels': ['first']},
            ).write(first)
            AffineStack(
                matrices=[AffineMatrix.from_translation([2., 0.])],
                metadata={'labels': ['second']},
            ).write(second)

            with patch.object(sys, 'argv', ['append_affine_stack', str(first), str(second), str(target)]):
                append_affine_stack()

            result = AffineStack.read(target)
            self.assertEqual(len(result), 2)
            np.testing.assert_allclose(result[0].translation, [1., 0.])
            np.testing.assert_allclose(result[1].translation, [2., 0.])
            self.assertEqual(result.get_metadata('labels'), ['first', 'second'])
