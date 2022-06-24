# Plant cataloging workflow

Source code for the plant cataloging workflow introduced in the paper

Günder, M. et al.: "Agricultural Plant Cataloging and Establishment of a Data Framework from UAV-based Crop Images by Computer Vision" [Publication at GigaScience](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giac054/6610009).
  
<details><summary>Citation</summary>
<p>

```bibtex
@article{10.1093/gigascience/giac054,
        author = {Günder, Maurice and Ispizua Yamati, Facundo R and Kierdorf, Jana and Roscher, Ribana and Mahlein, Anne-Katrin and Bauckhage, Christian},
        title = "{Agricultural plant cataloging and establishment of a data framework from UAV-based crop images by computer vision}",
        journal = {GigaScience},
        volume = {11},
        year = {2022},
        month = {06},
        abstract = "{Unmanned aerial vehicle (UAV)–based image retrieval in modern agriculture enables gathering large amounts of spatially referenced crop image data. In large-scale experiments, however, UAV images suffer from containing a multitudinous amount of crops in a complex canopy architecture. Especially for the observation of temporal effects, this complicates the recognition of individual plants over several images and the extraction of relevant information tremendously.In this work, we present a hands-on workflow for the automatized temporal and spatial identification and individualization of crop images from UAVs abbreviated as “cataloging” based on comprehensible computer vision methods. We evaluate the workflow on 2 real-world datasets. One dataset is recorded for observation of Cercospora leaf spot—a fungal disease—in sugar beet over an entire growing cycle. The other one deals with harvest prediction of cauliflower plants. The plant catalog is utilized for the extraction of single plant images seen over multiple time points. This gathers a large-scale spatiotemporal image dataset that in turn can be applied to train further machine learning models including various data layers.The presented approach improves analysis and interpretation of UAV data in agriculture significantly. By validation with some reference data, our method shows an accuracy that is similar to more complex deep learning–based recognition techniques. Our workflow is able to automatize plant cataloging and training image extraction, especially for large datasets.}",
        issn = {2047-217X},
        doi = {10.1093/gigascience/giac054},
        url = {https://doi.org/10.1093/gigascience/giac054},
        note = {giac054},
        eprint = {https://academic.oup.com/gigascience/article-pdf/doi/10.1093/gigascience/giac054/44118901/giac054.pdf},
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

* ### Installation as conda environment (Linux, Windows, and MacOS)

    Make sure, that ```conda``` is installed on your system. Install instructions for Miniconda can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Then, create and activate a new conda environment by

    ```bash
    $ conda env create -f environment.yml
    $ conda activate pcw
    ```

    The workflow package can now be used in this environment.
* ### Installation as Docker container (Linux only)

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

Make sure that ...
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
