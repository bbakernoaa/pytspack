from setuptools import setup, Extension
import numpy
import os

# Define compiler flags
extra_args = ["-std=c99", "-O3"]
if os.name != "nt":
    extra_args.append("-fPIC")

libpytspack = Extension(
    name="pytspack._libpytspack",
    sources=[
        "src/tspack.c",
        "src/pytspack.c",
    ],
    include_dirs=["src", numpy.get_include()],
    extra_compile_args=extra_args,
)

setup(
    ext_modules=[libpytspack],
    packages=["pytspack"],
)
