# Plant cataloging workflow

Source code for the plant cataloging workflow introduced in the paper "Agricultural Plant Cataloging and Establishment of a Data Framework from UAV-based Crop Images by Computer Vision"

Pre-print version avaliable on [arXiv](https://arxiv.org/abs/2201.02885v2).

<details><summary>Toggle citation</summary>
<p>

```bibtex
@misc{guender_plantcatalogingworkflow,
      title={Agricultural Plant Cataloging and Establishment of a Data Framework from UAV-based Crop Images by Computer Vision}, 
      author={Maurice G\"under and Facundo R. Ispizua Yamati and Jana Kierdorf and Ribana Roscher and Anne-Katrin Mahlein and Christian Bauckhage},
      year={2022},
      eprint={2201.02885},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

</p>
</details>

---
## Installation instructions

First, clone the repository and navigate to it

```bash
$ git clone https://github.com/mrcgndr/plant_cataloging_workflow.git
$ cd plant_cataloging_workflow
```

Two installation methods are recommended: By using a virtual environment or using Docker.

* ### Installation as ```conda``` environment (Linux and Windows)

    Make sure, that ```conda``` is installed on your system. Install instructions for Miniconda can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html). Then, create and activate a new conda environment by

    ```bash
    $ conda env create -f environment.yml
    $ conda activate pcw
    ```

    The workflow package can now be used in this environment.
* ### Installation as Docker Container (Linux only)

  Build Container with [```Dockerfile```](Dockerfile)

    ```bash
    $ docker build . -t pcw:latest
    ```

  Inlcude data and results folder as shared folders and start container image

    ```bash
    $ docker run -it --name pcw -v /local/path/to/data/:/root/pcw/data -v /local/path/to/results/:/root/pcw/results pcw:latest
    ```

---
## Start demo pipeline

Download the demo dataset [here](https://dx.doi.org/10.5524/102225) and make sure that ...
  * ... UAV image data located in ```data/``` folder
  * ... ground truth data located in ```data/ground_truth/``` folder

or change corresponding paths in the [demo config file](config/demo.yml).

Execute the demo pipeline with

```bash
$ python scripts/run_cataloging.py -c config/demo.yml
```

The results will be saved in  ```results/``` folder.

---
## Directory contents

* [```config```](/config): Pipeline configurations for plant extraction workflow in YAML format
* [```cataloging```](/cataloging): Actual code of plant cataloging workflow implementation
* [```scripts```](/scripts): Scripts to execute drone image analysis workflows.

---
## Declaration of dependencies

The Python implementation comes with some dependencies:

* Workflow, pipeline, and configuration
  * [```fluidml```](https://github.com/fluidml/fluidml) for pipeline
  * [```metadict```](https://github.com/LarsHill/metadict) and [```pyyaml```](https://pyyaml.org) for configuration handling
* File handling of results
  * [```h5py```](https://www.h5py.org) for image datasets
  * [```matplotlib```](https://matplotlib.org) for plotting
  * [```pandas```](https://pandas.pydata.org) for plant catalog
  * [```pykml```](https://pythonhosted.org/pykml/) and [```simplekml```](https://simplekml.readthedocs.io/en/latest/) for handling the georeferenced annotation files
* Orthoimage manipulation and IO
  * [```pyproj```](https://pyproj4.github.io/pyproj/stable/) for coordinate system transformations
  * [```rasterio```](https://rasterio.readthedocs.io/en/latest/) for reading/writing georeferenced images (orthoimages)
* Algorithms
  * [```pycpd```](https://github.com/siavashk/pycpd) for point set registration used in workflow
  * [```scikit-image```](https://scikit-image.org) for image processing and filtering
  * [```scikit-learn```](https://scikit-learn.org) for basic models
  * [```scipy```](https://scipy.org) for mathematical operations and common algorithms (peak finder, image transforms, etc.)
