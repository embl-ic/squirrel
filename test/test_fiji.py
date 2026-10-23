import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from squirrel.library.fiji import sift_log_line_to_affine, sift_log_to_affine_stack
from squirrel.workflows.fiji import sift_log_to_affine_stack_workflow


class TestFijiSiftImporter(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_sift_log_line_to_affine_inverse(self):
        print('Testing Fiji SIFT matrix conversion and inversion ...')
        line = 'Transformation Matrix: 1 0 4 0 1 7'
        matrix = sift_log_line_to_affine(line, pivot=[10., 20.])

        np.testing.assert_allclose(matrix.linear, np.eye(2))
        np.testing.assert_allclose(matrix.translation, [-7., -4.])
        np.testing.assert_allclose(matrix.pivot, [10., 20.])

    def test_sift_log_to_affine_stack(self):
        print('Testing Fiji SIFT stack parsing ...')
        lines = ['Other log line\n', 'Transformation Matrix: 1 0 4 0 1 7\n']
        stack = sift_log_to_affine_stack(lines, pivot=[10., 20.])

        self.assertEqual(len(stack), 1)
        self.assertFalse(stack.sequenced)
        np.testing.assert_allclose(stack[0].translation, [-7., -4.])

    def test_sift_log_to_affine_stack_workflow(self):
        print('Testing Fiji SIFT workflow and JSON serialization ...')
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / 'fiji.txt'
            output_path = Path(directory) / 'fiji.json'
            log_path.write_text('Transformation Matrix: 1 0 4 0 1 7\n', encoding='utf-8')

            stack = sift_log_to_affine_stack_workflow(
                str(log_path), out_filepath=str(output_path), pivot=[10., 20.]
            )

            self.assertEqual(len(stack), 2)
            self.assertFalse(stack.sequenced)
            np.testing.assert_allclose(stack[0].as_compact(), [[1., 0., 0.], [0., 1., 0.]])
            np.testing.assert_allclose(stack[1].translation, [-7., -4.])
            self.assertEqual(json.loads(output_path.read_text())['pivot'], [10., 20.])
