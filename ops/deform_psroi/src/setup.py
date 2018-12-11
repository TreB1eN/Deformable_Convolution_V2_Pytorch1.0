from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='modulated_deform_psroi_cuda',
    ext_modules=[
        CUDAExtension('modulated_deform_psroi_cuda', [
            'deformable_psroi_pooling_cuda.cpp',
            'deformable_psroi_pooling_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
