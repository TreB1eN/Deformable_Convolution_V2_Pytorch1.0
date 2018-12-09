from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='modulated_deform_conv_cuda',
    ext_modules=[
        CUDAExtension('modulated_deform_conv_cuda', [
            'modulated_deform_conv_cuda.cpp',
            'modulated_deform_conv_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
