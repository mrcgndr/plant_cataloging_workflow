import time

_this_year = time.strftime("%Y")
__version__ = '0.2'
__author__ = 'Maurice Günder'
__author_email__ = 'mguender@uni-bonn.de'
__license__ = ''
__copyright__ = f'Copyright (c) 2020-{_this_year}, {__author__}.'
__homepage__ = ''
__docs__ = ''

from . import config #import PipelineConfig
from . import pipeline #import Pipeline
from . import plant_images #import PlantImage, PlantImageStack
from . import tasks #import SegmentSoilPlants, FitGrowFunction, ExtractPlantPositions,\
                   #LoadPeaks, AlignPoints, AlignCroplines, FindCroplines, FilterWeed,\
                   #GroupPoints, SortGroupLabels, SavePlantsDataFrame, EvaluateDetectionQuality,\
                   #MergePlantsDataFrame, MakeImageDataset, MergeBoundingBoxFiles, ExportPlantPositions
from . import vi #import osaviImage, ngrdiImage, gliImage