from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nsection_cuda',
    ext_modules=[
        CUDAExtension('nsection_cuda', [
            'nsection_cuda.cpp',
            'nsection_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })