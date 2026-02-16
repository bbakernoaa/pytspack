from setuptools import setup, Extension
import numpy
import os

# Define compiler flags
if os.name == "nt":
    extra_args = ["/O2"]
    macros = [("BUILDING_TSPACK", "1")]
else:
    extra_args = ["-std=c99", "-O3", "-fPIC"]
    macros = []

libpytspack = Extension(
    name="pytspack._libpytspack",
    sources=[
        "src/tspack.c",
        "src/pytspack.c",
    ],
    include_dirs=["src", numpy.get_include()],
    extra_compile_args=extra_args,
    define_macros=macros,
)

setup(
    ext_modules=[libpytspack],
    packages=["pytspack"],
)
