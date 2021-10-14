# Plant extraction workflow

Source code for the plant extraction workflow introduced in the paper "Agricultural Plant Cataloging and Establishment of a Data Framework from UAV-based Crop Images by Computer Vision"

## Installation

### 1. Create new Python environment and activate it

Tested with Python 3.9.4+
```bash
$ python -m venv .venv
$ source .venv/bin/activate
```

### 2. Install requirements and package

```bash
$ pip install -U pip
$ pip install -r requirements.txt
$ python setup.py install
```

### Start demo workflow

* UAV image data located in data/ folder
* ground truth data located in data/ground_truth/ folder

```bash
$ scripts/run_plantextraction.py -c config/demo.yml
```

* results will be saved in results/ folder

## Directory contents

* [config](/config): Pipeline configurations for plant extraction workflow in YAML format
* [data](/data): Folder with UAV images and ground truth data
* [plant_extraction](/plant_extraction): Actual code of plant extraction workflow implementation
* [scripts](/scripts): Scripts to execute drone image analysis workflows.
