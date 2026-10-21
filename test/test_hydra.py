import unittest
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

import tifffile

from squirrel.library.hydra import (
    read_hydra_metadata, extract_hydra_metadata,
    HYDRA_METADATA_FIELDS, HYDRA_METADATA_PRESETS,
)


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


class TestHydraNamedMetadata(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)
        self.tmpdir = TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.path = Path(self.tmpdir.name) / 'sample.tif'
        xml = """<Metadata>
            <BinaryResult><PixelSize><X>7e-9</X><Y>8e-9</Y></PixelSize></BinaryResult>
            <StageSettings><StagePosition><X>0.001</X><Y>0.002</Y></StagePosition></StageSettings>
            <Optics><SamplePreTiltAngle>-0.5</SamplePreTiltAngle></Optics>
            <CustomSectionGroup>
              <CustomSection scope="VoxelSizeInfo">
                <VoxelSizeInfo><VoxelZ>15 nm</VoxelZ></VoxelSizeInfo>
              </CustomSection>
            </CustomSectionGroup>
        </Metadata>"""
        tifffile.imwrite(self.path, [[1, 2], [3, 4]], extratags=[
            (0x877B, 's', 0, xml, False),
        ])

    def test_extract_hydra_metadata_resolution(self):
        print('Testing extract_hydra_metadata: resolution preset and units')
        result = extract_hydra_metadata(self.path, preset='resolution')
        self.assertAlmostEqual(result['pixel_size_x']['value'], 7.0)
        self.assertAlmostEqual(result['pixel_size_y']['value'], 8.0)
        self.assertEqual(result['slice_thickness'], {'value': 15.0, 'unit': 'nm'})

    def test_extract_hydra_metadata_combined_selection(self):
        print('Testing extract_hydra_metadata: combine preset and fields')
        result = extract_hydra_metadata(
            self.path, preset='resolution', fields=['stage_x', 'pixel_size_x']
        )
        self.assertEqual(list(result), [
            'pixel_size_x', 'pixel_size_y', 'slice_thickness', 'stage_x'
        ])
        self.assertEqual(result['stage_x'], {'value': 0.001, 'unit': 'm'})

    def test_extract_hydra_metadata_missing_field(self):
        print('Testing extract_hydra_metadata: missing known field')
        xml = '<Metadata><BinaryResult /></Metadata>'
        tifffile.imwrite(self.path, [[1, 2], [3, 4]], extratags=[
            (0x877B, 's', 0, xml, False),
        ])
        result = extract_hydra_metadata(self.path, fields=['slice_thickness'])
        self.assertEqual(result['slice_thickness'], {'value': None, 'unit': 'nm'})

    def test_extract_hydra_metadata_unknown_field_and_preset(self):
        print('Testing extract_hydra_metadata: reject unknown names')
        with self.assertRaisesRegex(ValueError, 'Unknown Hydra metadata field'):
            extract_hydra_metadata(self.path, fields=['wrong'])
        with self.assertRaisesRegex(ValueError, 'Unknown Hydra metadata preset'):
            extract_hydra_metadata(self.path, preset='wrong')

    def test_extract_hydra_metadata_invalid_unit(self):
        print('Testing extract_hydra_metadata: reject unexpected units')
        xml = ('<Metadata><CustomSectionGroup>'
               '<CustomSection scope="VoxelSizeInfo"><VoxelSizeInfo>'
               '<VoxelZ>15 um</VoxelZ></VoxelSizeInfo></CustomSection>'
               '</CustomSectionGroup></Metadata>')
        tifffile.imwrite(self.path, [[1, 2], [3, 4]], extratags=[
            (0x877B, 's', 0, xml, False),
        ])
        with self.assertRaisesRegex(ValueError, 'Unexpected unit'):
            extract_hydra_metadata(self.path, fields=['slice_thickness'])
