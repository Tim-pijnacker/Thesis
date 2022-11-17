from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='nsection_cpp',
      ext_modules=[cpp_extension.CppExtension('nsection_cpp', ['nsection.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})