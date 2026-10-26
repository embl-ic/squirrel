import unittest
import warnings
from unittest.mock import patch


class FakeParameterMap(dict):
    pass


class TestElastixStackCollection(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)
        import sys
        from types import ModuleType

        fake = ModuleType('SimpleITK')
        fake.ParameterMap = FakeParameterMap
        self.patcher = patch.dict(sys.modules, {'SimpleITK': fake})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    @staticmethod
    def make_map(value):
        return FakeParameterMap({'Size': ['8', '9'], 'TransformParameters': [str(value)]})

    def test_empty_and_slicing(self):
        print('Testing ElastixStack: empty construction and slicing')
        from squirrel.library.elastix import ElastixStack

        stack = ElastixStack()
        self.assertEqual(len(stack), 0)
        self.assertIsInstance(stack[:], ElastixStack)
        stack.extend([self.make_map(1), self.make_map(2)])
        self.assertIsInstance(stack[0], FakeParameterMap)
        self.assertIsInstance(stack[:1], ElastixStack)
        self.assertEqual(len(stack[:1]), 1)

    def test_slice_is_independent(self):
        print('Testing ElastixStack: slicing copies parameter maps')
        from squirrel.library.elastix import ElastixStack

        stack = ElastixStack(transforms=[self.make_map(1), self.make_map(2)])
        sliced = stack[:1]
        sliced[0]['TransformParameters'] = ['99']

        self.assertEqual(stack[0]['TransformParameters'], ['1'])
        self.assertEqual(len(sliced), 1)

    def test_copy_and_independent_iteration(self):
        print('Testing ElastixStack: copy isolation and nested iteration')
        from squirrel.library.elastix import ElastixStack

        stack = ElastixStack(transforms=[self.make_map(1), self.make_map(2)])
        copied = stack.copy()
        copied[0]['TransformParameters'] = ['99']
        self.assertEqual(stack[0]['TransformParameters'], ['1'])
        pairs = [(a, b) for a in stack for b in stack]
        self.assertEqual(len(pairs), 4)

    def test_append_extend_and_constructor_validation(self):
        print('Testing ElastixStack: append, extend and constructor validation')
        from squirrel.library.elastix import ElastixStack

        stack = ElastixStack(transforms=[self.make_map(1)])
        with self.assertRaises(TypeError):
            stack.append(ElastixStack())
        stack.extend(ElastixStack(transforms=[self.make_map(2)]))
        self.assertEqual(len(stack), 2)
        with self.assertRaises(TypeError):
            ElastixStack(transforms=[object()])

    def test_extend_rejects_invalid_input_without_partial_mutation(self):
        print('Testing ElastixStack: extend validates before modifying the stack')
        from squirrel.library.elastix import ElastixStack

        stack = ElastixStack(transforms=[self.make_map(1)])
        with self.assertRaises(TypeError):
            stack.extend([self.make_map(2), object()])
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack[0]['TransformParameters'], ['1'])

        with self.assertRaises(TypeError):
            stack.extend(None)
        self.assertEqual(len(stack), 1)

    def test_extend_copies_maps_and_accepts_generator(self):
        print('Testing ElastixStack: extend copies maps from a generator')
        from squirrel.library.elastix import ElastixStack

        source = [self.make_map(1), self.make_map(2)]
        stack = ElastixStack()
        stack.extend(transform for transform in source)
        source[0]['TransformParameters'][0] = '99'

        self.assertEqual(len(stack), 2)
        self.assertEqual(stack[0]['TransformParameters'], ['1'])
