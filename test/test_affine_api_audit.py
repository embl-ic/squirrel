import ast
import unittest
import warnings
from pathlib import Path

import numpy as np

from squirrel.library.affine_matrices import AffineMatrix, AffineStack


class TestAffineApiAudit(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_removed_affine_api_is_not_used(self):
        print('Testing source modules for removed affine API calls')
        root = Path(__file__).resolve().parents[1] / 'squirrel'
        removed_methods = {
            'to_file', 'get_matrix', 'get_sequenced_stack',
            'get_median_smoothed_stack', 'get_smoothed_stack',
            'get_translation', 'get_scaled', 'get_meta', 'set_meta',
        }
        removed_keywords = {
            'AffineMatrix': {'filepath', 'parameters', 'translation', 'elastix_parameters'},
            'AffineStack': {'filepath', 'stack', 'is_sequenced', 'pivot'},
        }
        violations = []

        for path in sorted(root.rglob('*.py')):
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute) and node.func.attr in removed_methods:
                        violations.append(f'{path.relative_to(root)}:{node.lineno}: {node.func.attr}')
                    if isinstance(node.func, ast.Name) and node.func.id in removed_keywords:
                        for keyword in node.keywords:
                            if keyword.arg in removed_keywords[node.func.id]:
                                violations.append(f'{path.relative_to(root)}:{node.lineno}: {keyword.arg}')
                if isinstance(node, ast.Attribute) and node.attr == 'is_sequenced':
                    violations.append(f'{path.relative_to(root)}:{node.lineno}: is_sequenced')

        self.assertEqual(violations, [], '\n'.join(violations))

    def test_affine_composition_order(self):
        print('Testing affine composition order for noncommuting matrices')
        left = AffineMatrix([[2., 0., 3.], [0., 1., 4.], [0., 0., 1.]])
        right = AffineMatrix([[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]])
        composed = left @ right
        np.testing.assert_allclose(composed.as_array(), left.as_array() @ right.as_array())
        self.assertFalse(np.allclose(composed.as_array(), (right @ left).as_array()))

    def test_median_trend_composition_and_input_immutability(self):
        print('Testing median trend removal with noncommuting affine matrices')
        matrices = [
            AffineMatrix([[1., float(index) / 10, float(index)],
                          [0., 1., float(index % 3)], [0., 0., 1.]])
            for index in range(7)
        ]
        transforms = AffineStack(matrices=matrices, sequenced=False, metadata={'source': 'audit'})
        original = transforms.as_array().copy()
        smoothed = transforms.smooth_median(2)
        result = transforms @ smoothed.inverse()

        for index in range(len(transforms)):
            expected = transforms[index].as_array() @ smoothed[index].inverse().as_array()
            np.testing.assert_allclose(result[index].as_array(), expected)
        np.testing.assert_allclose(transforms.as_array(), original)
        self.assertEqual(result.metadata, transforms.metadata)
        self.assertEqual(result.sequenced, transforms.sequenced)
