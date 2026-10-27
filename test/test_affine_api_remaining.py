import unittest
import warnings
from unittest.mock import patch

import numpy as np

from squirrel.library.affine_matrices import AffineMatrix, AffineStack
from squirrel.library.template_matching import match_template_on_image
from squirrel.workflows.template_matching import template_matching_stack_alignment_workflow


class TestRemainingAffineApi(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_template_match_returns_translation(self):
        print('Testing template matching affine construction')
        image = np.zeros((12, 12), dtype=float)
        template = np.array([[0., 1., 0.], [1., 3., 1.], [0., 1., 0.]])
        image[4:7, 6:9] = template
        result = match_template_on_image(image, template)
        self.assertIsInstance(result, AffineMatrix)
        np.testing.assert_allclose(result.translation, [4., 6.])

    def test_template_matching_workflow_writes_metadata(self):
        print('Testing template matching stack serialization and metadata')
        import tempfile
        from pathlib import Path
        stack = np.zeros((2, 12, 12), dtype=float)
        template = np.array([[0., 1., 0.], [1., 3., 1.], [0., 1., 0.]])
        stack[:, 4:7, 6:9] = template
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / 'transforms.json')
            with patch('squirrel.library.io.load_data_handle', return_value=(stack, stack.shape)):
                template_matching_stack_alignment_workflow(
                    stack='unused', out_filepath=output,
                    template_roi=[4, 6, 7, 9, 0],
                    z_range=[0, 2], n_threads=1,
                )
            result = AffineStack.read(output)
            self.assertTrue(result.sequenced)
            self.assertEqual(len(result), 2)
            self.assertTrue(result.has_metadata('bounds'))
            np.testing.assert_allclose(result[0].translation, [0., 0.])
