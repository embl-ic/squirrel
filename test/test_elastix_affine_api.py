import unittest
import warnings
from unittest.mock import MagicMock, patch

import numpy as np

from squirrel.library.affine_matrices import AffineMatrix, AffineStack
from squirrel.library.elastix import apply_transforms_on_image


class TestElastixAffineApi(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_affine_stack_constructor_and_pivot(self):
        print('Testing Elastix affine stack construction and pivot')
        stack = AffineStack(sequenced=True)
        stack.append(AffineMatrix.identity(ndim=2, pivot=[0., 0.]))
        self.assertTrue(stack.sequenced)
        np.testing.assert_array_equal(stack.pivot, [0., 0.])

    def test_affine_scaling_preserves_translation(self):
        print('Testing Elastix initialization scaling')
        matrix = AffineMatrix.from_translation([2., -3.])
        scaled = matrix.scaled_for_image_resize(4)
        np.testing.assert_allclose(scaled.translation, [8., -12.])

    def test_transformix_receives_parameter_map(self):
        print('Testing Elastix image application parameter map conversion')
        import sys

        fake_sitk = MagicMock()
        fake_sitk.Image = type('Image', (), {})
        fake_sitk.ParameterMap.return_value = {}
        fake_sitk.GetArrayFromImage.return_value = np.zeros((4, 4), dtype='uint8')
        image = np.zeros((4, 4), dtype='uint8')
        matrix = AffineMatrix.from_translation([1., 2.])

        with patch.dict(sys.modules, {'SimpleITK': fake_sitk}):
            apply_transforms_on_image(image, [matrix], n_workers=1)

        parameter_map = fake_sitk.TransformixImageFilter.return_value.AddTransformParameterMap.call_args.args[0]
        self.assertEqual(parameter_map['Transform'], ['AffineTransform'])
        self.assertEqual(parameter_map['Size'], ['4', '4'])
