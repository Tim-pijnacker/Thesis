from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='hello_cpp',
      ext_modules=[cpp_extension.CppExtension('hello_cpp', ['hello.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})