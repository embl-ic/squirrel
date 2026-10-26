import os
import tempfile
import unittest
import warnings
from unittest.mock import patch

from test_elastix_stack_collection import FakeParameterMap


class TestElastixStackIO(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)
        import sys
        from types import ModuleType

        fake = ModuleType('SimpleITK')
        fake.ParameterMap = FakeParameterMap

        def read_parameter_file(filepath):
            with open(filepath, encoding='utf-8') as file:
                return FakeParameterMap({'TransformParameters': [file.read()]})

        def write_parameter_file(transform, filepath):
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(transform['TransformParameters'][0])

        fake.ReadParameterFile = read_parameter_file
        fake.WriteParameterFile = write_parameter_file
        self.patcher = patch.dict(sys.modules, {'SimpleITK': fake})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_read_write_roundtrip(self):
        print('Testing ElastixStack: read/write roundtrip')
        from squirrel.library.elastix import ElastixStack

        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, 'transforms')
            stack = ElastixStack(transforms=[FakeParameterMap({'TransformParameters': ['7']})])
            stack.write(path)
            restored = ElastixStack.read(path)
            self.assertEqual(len(restored), 1)
            self.assertEqual(restored[0]['TransformParameters'], ['7'])

            other_path = os.path.join(root, 'legacy')
            restored.write(other_path)
            self.assertEqual(len(ElastixStack.read(other_path)), 1)

    def test_read_many_and_legacy_helper(self):
        print('Testing ElastixStack: read_many ordering and legacy helper')
        from squirrel.library.elastix import ElastixStack, load_transform_stack_from_multiple_files

        with tempfile.TemporaryDirectory() as root:
            paths = []
            for idx in range(2):
                path = os.path.join(root, str(idx))
                ElastixStack(transforms=[FakeParameterMap({'TransformParameters': [str(idx)]})]).write(path)
                paths.append(path)

            combined = ElastixStack.read_many(paths)
            self.assertEqual([t['TransformParameters'][0] for t in combined], ['0', '1'])
            self.assertEqual(len(load_transform_stack_from_multiple_files(paths)), 2)
            self.assertEqual(len(ElastixStack.read_many([])), 0)

    def test_read_pattern(self):
        print('Testing ElastixStack: read file pattern')
        from squirrel.library.elastix import ElastixStack

        with tempfile.TemporaryDirectory() as root:
            for name in ('a.txt', 'b.map'):
                with open(os.path.join(root, name), 'w', encoding='utf-8') as file:
                    file.write(name)
            self.assertEqual(len(ElastixStack.read(root)), 1)
            self.assertEqual(len(ElastixStack.read(root, pattern='*.map')), 1)
