from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='kivi_gemv',
    ext_modules=[
        CUDAExtension(
            name='kivi_gemv',
            sources=['pybind.cpp', 'gemv_cuda.cu'],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)