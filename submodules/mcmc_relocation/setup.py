from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="mcmc_relocation",
    packages=['mcmc_relocation'],
    ext_modules=[
        CUDAExtension(
            name="mcmc_relocation._C",
            sources=[
                "ext.cpp",
                "mcmc_relocation.cu",
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
