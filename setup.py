from setuptools import setup, Extension
import numpy
import os

# Define compiler flags
extra_args = ["-std=c99", "-O3"]
if os.name != "nt":
    extra_args.append("-fPIC")

librenka = Extension(
    name="renka._librenka",
    sources=[
        "src/tspack.c",
        "src/stripack.c",
        "src/ssrfpack.c",
        "src/srfpack.c",
        "src/tripack.c",
        "src/renka.c",
    ],
    include_dirs=["src", numpy.get_include()],
    extra_compile_args=extra_args,
)

setup(
    ext_modules=[librenka],
    packages=["renka"],
)
