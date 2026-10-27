import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

from squirrel.library.affine_matrices import AffineMatrix, AffineStack
from squirrel.workflows.elastix import apply_multi_step_stack_alignment_workflow


class TestMultiStepAffineSequencing(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_relative_stack_sequenced_before_substack(self):
        print('Testing multi-step alignment: sequence before selecting a substack')

        class FakeElastixStack:
            def __init__(self, transforms, image_shape):
                self.transforms = transforms

        class FakeMultiStepStack:
            def __init__(self, stacks, image_shape):
                self.stacks = stacks

            def apply_on_image_stack(self, *args, **kwargs):
                return self.stacks[0].transforms

        transforms = AffineStack(
            matrices=[AffineMatrix.from_translation([1., 0.]) for _ in range(4)],
            sequenced=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / 'transforms.json')
            transforms.write(path)

            with patch('squirrel.library.io.load_data_handle', return_value=(None, (4, 8, 8))), \
                    patch('squirrel.library.elastix.ElastixStack', FakeElastixStack), \
                    patch('squirrel.library.elastix.ElastixMultiStepStack', FakeMultiStepStack):
                result = apply_multi_step_stack_alignment_workflow(
                    'unused', [path], z_range=(2, 4), start_transform_id=2,
                )

                self.assertTrue(result.sequenced)
                np.testing.assert_allclose(
                    [matrix.translation[0] for matrix in result], [3., 4.]
                )

                with self.assertRaisesRegex(AssertionError, 'not sequenced'):
                    apply_multi_step_stack_alignment_workflow(
                        'unused', [path], assert_sequenced=True,
                    )
