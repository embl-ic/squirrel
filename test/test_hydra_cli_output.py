import io
import sys
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import tifffile
import yaml


class TestHydraCliOutput(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)
        self.tmpdir = TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.path = Path(self.tmpdir.name) / 'sample.tif'
        xml = ('<Metadata><BinaryResult><PixelSize><X>7e-9</X>'
               '<Y>8e-9</Y></PixelSize></BinaryResult></Metadata>')
        tifffile.imwrite(self.path, [[1, 2], [3, 4]], extratags=[
            (0x877B, 's', 0, xml, False),
        ])

    def test_extract_tif_metadata_stdout(self):
        print('Testing extract_tif_metadata: stdout is the default')
        from squirrel.hydra import extract_tif_metadata

        for options in ([], ['--fields', 'pixel_size_x']):
            with self.subTest(options=options):
                stdout = io.StringIO()
                argv = ['sq-hydra-extract_tif_metadata', str(self.path)] + options
                with patch.object(sys, 'argv', argv), redirect_stdout(stdout):
                    extract_tif_metadata()
                result = yaml.safe_load(stdout.getvalue())
                if options:
                    self.assertAlmostEqual(result['pixel_size_x']['value'], 7.0)
                else:
                    self.assertIn('Metadata', result)

    def test_extract_tif_metadata_file_options(self):
        print('Testing extract_tif_metadata: -o and --out_filepath')
        from squirrel.hydra import extract_tif_metadata

        for flag in ('-o', '--out_filepath'):
            with self.subTest(flag=flag):
                output = Path(self.tmpdir.name) / 'result.yaml'
                argv = ['sq-hydra-extract_tif_metadata', str(self.path),
                        '--fields', 'pixel_size_x', flag, str(output)]
                with patch.object(sys, 'argv', argv):
                    extract_tif_metadata()
                result = yaml.safe_load(output.read_text(encoding='utf-8'))
                self.assertAlmostEqual(result['pixel_size_x']['value'], 7.0)
                output.unlink()
