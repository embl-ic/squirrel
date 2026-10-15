import unittest
import warnings
from pathlib import Path


class TestNormalizationDocumentation(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)
        self.root = Path(__file__).resolve().parents[1]

    def test_filters_documentation_normalization_options(self):
        print('Testing filters documentation: normalization options are documented')
        content = (self.root / 'docs' / 'filters.md').read_text()
        for option in ('normalization_method', 'quantiles', 'anchors',
                       'normalization_range', 'keep_zeros', 'ignore_zeros'):
            with self.subTest(option=option):
                self.assertIn(option, content)

    def test_readme_links_to_filter_documentation(self):
        print('Testing README: filter documentation link exists')
        readme = (self.root / 'README.md').read_text()
        self.assertIn('docs/filters.md', readme)
        self.assertTrue((self.root / 'docs' / 'filters.md').is_file())
