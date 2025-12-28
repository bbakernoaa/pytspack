from setuptools import setup, Extension

ext = Extension(
    name="pytspack.tspack",
    sources=["pytspack/tspack.c"],
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
