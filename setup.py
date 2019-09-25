from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='global_contrast',
    ext_modules=[
        CUDAExtension('global_contrast', [
            'global_contrast.cpp',
            'global_contrast_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

