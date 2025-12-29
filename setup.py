from setuptools import setup, Extension
import numpy

# Set up the extension module
ext = Extension(
    "renka.renka",  # Package renka, extension renka
    sources=[
        "renka/renka.c",
        "renka/srfpack.c",
        "renka/ssrfpack.c",
        "renka/stripack.c",
        "renka/tripack.c",
        "renka/tspack.c",
    ],
    include_dirs=[numpy.get_include(), "renka"],
    # Ensure C99/Math lib if needed
    extra_compile_args=["-O3", "-fPIC"],
)

setup(
    name="renka",
    version="0.1.0",
    description="Python interface to Renka's triangulation and spline packages (tspack, tripack, stripack, srfpack, ssrfpack)",
    packages=["renka"],
    ext_modules=[ext],
    setup_requires=["numpy"],
    install_requires=["numpy"],
)
