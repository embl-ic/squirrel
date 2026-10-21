import unittest
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

import tifffile

from squirrel.library.hydra import read_hydra_metadata


class TestHydraMetadata(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)
        self.tmpdir = TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _write_tiff(self, xml=None):
        path = Path(self.tmpdir.name) / 'sample.tif'
        extratags = []

        if xml is not None:
            encoded = xml.encode('utf-8') + b'\x00'
            extratags = [(0x877B, 'B', len(encoded), encoded, False)]

        import numpy as np
        tifffile.imwrite(path, np.zeros((4, 4), dtype='uint8'),
                         extratags=extratags)
        return path

    def test_read_hydra_metadata_nested_and_repeated_elements(self):
        print('Testing read_hydra_metadata: XML hierarchy and repeated tags')
        xml = ('<Metadata version="1"><PixelSize><X>5e-9</X></PixelSize>'
               '<Item name="a">first</Item><Item name="b">second</Item>'
               '<Empty /></Metadata>')
        path = self._write_tiff(xml)

        result = read_hydra_metadata(path)

        self.assertEqual(result['Metadata']['@version'], '1')
        self.assertEqual(result['Metadata']['PixelSize']['X'], '5e-9')
        self.assertEqual(result['Metadata']['Item'], [
            {'@name': 'a', '#text': 'first'},
            {'@name': 'b', '#text': 'second'}
        ])
        self.assertIsNone(result['Metadata']['Empty'])

    def test_read_hydra_metadata_missing_tag(self):
        print('Testing read_hydra_metadata: missing FEI tag')
        path = self._write_tiff()

        with self.assertRaisesRegex(ValueError, '0x877B'):
            read_hydra_metadata(path)

    def test_read_hydra_metadata_invalid_xml(self):
        print('Testing read_hydra_metadata: malformed XML')
        path = self._write_tiff('<Metadata><Broken></Metadata>')

        with self.assertRaises(ET.ParseError):
            read_hydra_metadata(path)
