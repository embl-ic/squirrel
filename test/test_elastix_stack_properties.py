import unittest
import warnings
from unittest.mock import patch
import sys
from types import ModuleType

import numpy as np

from test_elastix_stack_collection import FakeParameterMap


class TestElastixStackProperties(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)
        fake = ModuleType('SimpleITK')
        fake.ParameterMap = FakeParameterMap
        self.patcher = patch.dict(sys.modules, {'SimpleITK': fake})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_image_shape_valid_and_empty(self):
        print('Testing ElastixStack: image_shape property and empty stacks')
        from squirrel.library.elastix import ElastixStack

        self.assertIsNone(ElastixStack().image_shape)
        stack = ElastixStack(transforms=[FakeParameterMap({'Size': ['12', '8']})] * 2)
        np.testing.assert_array_equal(stack.image_shape, [8, 12])

    def test_image_shape_invalid(self):
        print('Testing ElastixStack: image_shape validation')
        from squirrel.library.elastix import ElastixStack

        invalid = [
            [{'Size': ['12', '8']}, {'Size': ['12', '9']}],
            [{'Transform': ['AffineTransform']}],
            [{'Size': ['bad', '8']}],
            [{'Size': ['0', '8']}],
            [{'Size': ['12.5', '8']}],
            [{'Size': []}],
        ]
        for transforms in invalid:
            with self.subTest(transforms=transforms):
                stack = ElastixStack(transforms=[FakeParameterMap(item) for item in transforms])
                with self.assertRaises(ValueError):
                    _ = stack.image_shape

    def test_repr(self):
        print('Testing ElastixStack: representation')
        from squirrel.library.elastix import ElastixStack

        self.assertEqual(repr(ElastixStack()), 'ElastixStack(n_transforms=0)')
        stack = ElastixStack(transforms=[FakeParameterMap({'Size': ['2', '3']})])
        self.assertEqual(repr(stack), 'ElastixStack(n_transforms=1)')
