# Plant cataloging workflow

Source code for the plant cataloging workflow introduced in the paper "Agricultural Plant Cataloging and Establishment of a Data Framework from UAV-based Crop Images by Computer Vision"

Pre-print version avaliable on [arXiv](https://arxiv.org/abs/2201.02885v2).

Please cite as:
```bibtex
@misc{guender_plantcatalogingworkflow,
      title={Agricultural Plant Cataloging and Establishment of a Data Framework from UAV-based Crop Images by Computer Vision}, 
      author={Maurice Gu\"nder and Facundo R. Ispizua Yamati and Jana Kierdorf and Ribana Roscher and Anne-Katrin Mahlein and Christian Bauckhage},
      year={2022},
      eprint={2201.02885},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Installation with Virtual Environment

### 1. Create new Python environment and activate it

Tested with Python 3.9
```bash
$ python -m venv .venv
$ source .venv/bin/activate
```

### 2. Install requirements and package

Update pip
```bash
$ pip install -U pip
```

For installation as a regular package, run
```bash
$ pip install .
```

For installation in development mode, run
```bash
$ pip install -e .
```

## Execution with Docker

### 1. Build Container with [Dockerfile](Dockerfile)

```bash
$ docker build . -t pcw:latest
```

### 2. Run Container

Inlcude data and results folder as shared folders and start image

```bash
$ docker run -it -v /local/path/to/data/:/root/data -v /local/path/to/results/:/root/results pcw:latest
```

## Start demo pipeline

* UAV image data located in ```data/``` folder
* ground truth data located in ```data/ground_truth/``` folder

```bash
$ python scripts/run_cataloging.py -c config/demo.yml
```

* results will be saved in results/ folder

## Directory contents

* [config](/config): Pipeline configurations for plant extraction workflow in YAML format
* [cataloging](/cataloging): Actual code of plant cataloging workflow implementation
* [scripts](/scripts): Scripts to execute drone image analysis workflows.

## Declaration of dependencies

The Python implementation comes with some dependencies:

* Workflow, pipeline, and configuration
  * [fluidml](https://github.com/fluidml/fluidml) for pipeline
  * [metadict](https://github.com/LarsHill/metadict) and [pyyaml](https://pyyaml.org) for configuration handling
* File handling of results
  * [h5py](https://www.h5py.org) for image datasets
  * [matplotlib](https://matplotlib.org) for plotting
  * [pandas](https://pandas.pydata.org) for plant catalog
  * [pykml](https://pythonhosted.org/pykml/) and [simplekml](https://simplekml.readthedocs.io/en/latest/) for handling the georeferenced annotation files
* Orthoimage manipulation and IO
  * [pyproj](https://pyproj4.github.io/pyproj/stable/) for coordinate system transformations
  * [rasterio](https://rasterio.readthedocs.io/en/latest/) for reading/writing georeferenced images (orthoimages)
* Algorithms
  * [pycpd](https://github.com/siavashk/pycpd) for point set registration used in workflow
  * [scikit-image](https://scikit-image.org) for image processing and filtering
  * [scikit-learn](https://scikit-learn.org) for basic models
  * [scipy](https://scipy.org) for mathematical operations and common algorithms (peak finder, image transforms, etc.)
