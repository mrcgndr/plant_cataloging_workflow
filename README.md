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

## Installation

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

### Start demo pipeline

* UAV image data located in data/ folder
* ground truth data located in data/ground_truth/ folder

```bash
$ scripts/run_cataloging.py -c config/demo.yml
```

* results will be saved in results/ folder

## Directory contents

* [config](/config): Pipeline configurations for plant extraction workflow in YAML format
* [data](/data): Folder with UAV images and ground truth data
* [cataloging](/cataloging): Actual code of plant cataloging workflow implementation
* [scripts](/scripts): Scripts to execute drone image analysis workflows.
