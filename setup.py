import os
from setuptools import setup, Extension

# Get the NumPy include directory.
import numpy

ext = Extension(
    name="pytspack.tspack",
    sources=[
        "pytspack/tspack.c",
        "pytspack/tripack.c",
        "pytspack/stripack.c",
        "pytspack/srfpack.c",
        "pytspack/ssrfpack.c",
        "pytspack/common.c",
    ],
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-std=c99"],
)

if __name__ == "__main__":
    setup(
        name="pytspack",
        version="0.2.0",
        description="Wrapper around Robert J. Renka's C-translated TSPACK: Tension Spline Curve Fitting Package",
        author="Barry D. Baker",
        license="MIT",
        author_email="barry.baker@noaa.gov",
        packages=["pytspack"],
        ext_modules=[ext],
        install_requires=["numpy"],
    )
