from setuptools import setup, find_packages

from cataloging import __version__, __author__, __author_email__, __license__, __copyright__, __homepage__, __docs__

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(name='plant_cataloging_workflow',
      version=__version__,
      description=__docs__,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author=__author__,
      author_email=__author_email__,
      url=__homepage__,
      download_url=__homepage__,
      license=__license__,
      copyright=__copyright__,
      keywords=['remote-sensing', 'agriculture', 'image-processing', 'machine-learning'],
      packages=find_packages(),
      python_requires='>=3.9',
      install_requires=[
        "fluidml",
        "h5py",
        "matplotlib",
        "metadict",
        "pandas",
        "pycpd",
        "pykml",
        "pyproj",
        "pyyaml",
        "rasterio",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "simplekml"
      ])
