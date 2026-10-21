import unittest
import warnings
from unittest.mock import patch

import numpy as np


class TestFilterProgress(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)
        self.stack = np.arange(48, dtype=np.uint8).reshape(3, 4, 4)
        self.pipeline = [['same', {}]]

    def test_progress_all_backends(self):
        print('Testing progress: slice counts and unchanged output')
        from squirrel.library.filters import ImageFilter

        for backend, workers in [('serial', 1), ('threading', 2), ('multiprocessing', 2)]:
            with self.subTest(backend=backend):
                increments = []
                result = ImageFilter(self.stack).get_filtered_stack(
                    self.pipeline, n_workers=workers, backend=backend,
                    progress=False, progress_callback=increments.append)
                np.testing.assert_array_equal(result, self.stack)
                self.assertEqual(increments, [1, 1, 1])

    def test_progress_bar_optional(self):
        print('Testing progress: bar enabled and disabled')
        from squirrel.library.filters import ImageFilter

        with patch('tqdm.auto.tqdm') as bar:
            ImageFilter(self.stack).get_filtered_stack(self.pipeline, progress=False)
            bar.assert_not_called()

        with patch('tqdm.auto.tqdm') as bar:
            ImageFilter(self.stack).get_filtered_stack(self.pipeline, progress=True)
            bar.assert_called_once()
            self.assertEqual(bar.call_args.kwargs['total'], 3)
            self.assertEqual(bar.return_value.__enter__.return_value.update.call_count, 3)

    def test_progress_invalid_options(self):
        print('Testing progress: validation')
        from squirrel.library.filters import ImageFilter

        with self.assertRaisesRegex(TypeError, 'progress must be a boolean'):
            ImageFilter(self.stack).get_filtered_stack(self.pipeline, progress='yes')
        with self.assertRaisesRegex(TypeError, 'progress_callback must be callable'):
            ImageFilter(self.stack).get_filtered_stack(self.pipeline, progress_callback=5)
