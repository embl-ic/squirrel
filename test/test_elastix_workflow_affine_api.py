import unittest
import warnings

import numpy as np

from squirrel.library.affine_matrices import AffineMatrix, AffineStack


class TestElastixWorkflowAffineAPI(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_subtract_median_transform_trend(self):
        print('Testing median trend subtraction: constant translations cancel')

        transforms = AffineStack(
            matrices=[AffineMatrix.from_translation([3., -2.]) for _ in range(5)],
            sequenced=False,
            metadata={'source': 'test'},
        )
        corrected = transforms @ transforms.smooth_median(2).inverse()

        np.testing.assert_allclose(
            np.array([matrix.translation for matrix in corrected]),
            np.zeros((5, 2)), atol=1e-10,
        )
        self.assertEqual(corrected.metadata, transforms.metadata)
        self.assertFalse(corrected.sequenced)

    def test_replace_large_translations(self):
        print('Testing large translation replacement: outliers become identity')

        transforms = AffineStack(matrices=[
            AffineMatrix.from_translation([1., 0.]),
            AffineMatrix.from_translation([10., 0.]),
        ])
        corrected = transforms.replace_large_translations(5.)

        np.testing.assert_allclose(corrected[0].translation, [1., 0.])
        np.testing.assert_allclose(corrected[1].translation, [0., 0.])
        np.testing.assert_allclose(transforms[1].translation, [10., 0.])
