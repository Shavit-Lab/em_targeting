from setuptools import setup, find_packages

__version__ = "0.0.1"

setup(
    name="em_targeting",
    version=__version__,
    packages=["em_targeting"],
    author="Thomas Athey",
    author_email="tom.l.athey@gmail.com",
    install_requires=["napari", "numpy", "pandas", "Pillow", "scikit-image", "h5py"],
    license="MIT",
)
