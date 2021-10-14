from dataclasses import dataclass
from typing import List, Union

@dataclass
class FluidMLConfig():
    save_dir: str
    logging_level: str
    n_workers: int
    force: Union[List[str], None]
    visualize_graph: bool

@dataclass
class ImageConfig():
    source_dir: str
    img_format: str
    channels: List[str]
    field_ids: List[str]

@dataclass
class SegmentationConfig():
    veg_index: str
    use_watershed: bool
    use_growfunction_coverratio: bool

@dataclass
class PlantDetetionConfig():
    gauss_sigma_bounds: List[Union[float, int]]
    peak_threshold: float
    min_peak_distance: Union[float, int]
    max_detection_coverratio: float

@dataclass
class AlignmentConfig():
    max_centroid_distance: Union[float, int]

@dataclass
class WeedFilterConfig():
    threshold_factor: float

@dataclass
class PlantGroupingConfig():
    max_centroid_distance: Union[float, int]

@dataclass
class DataframesConfig():
    save_dir: str

@dataclass
class AnnotationsConfig():
    df_path: str
    gps_column: str
    values_column: str
    tol_distance: float

@dataclass
class DatasetGenerationConfig():
    shape: List[int]
    save_dir: str
    channels: List[str]
    annotations: AnnotationsConfig

@dataclass
class QualityEvaluationConfig():
    ground_truth_data_dir: Union[str, None]
    max_detection_distance: Union[float, int, None]

@dataclass
class OrthoimagesConfig():
    make_orthoimages: bool
    save_dir: str

@dataclass
class PlantDetectionPlotConfig():
    make_plots: bool
    img_format: str
    dpi: int
    cmap: str
    save_dir: str

@dataclass
class PlantGroupingPlotConfig():
    make_plots: bool
    img_format: str
    dpi: int
    cmap: str
    save_dir: str

@dataclass
class QualityEvaluationPlotConfig():
    make_plots: bool
    img_format: str
    dpi: int
    save_dir: str

@dataclass
class PlotConfig():
    plant_detection: PlantDetectionPlotConfig
    plant_grouping: PlantGroupingPlotConfig
    quality_evaluation: QualityEvaluationPlotConfig

@dataclass
class PipelineConfig():
    fluidml: FluidMLConfig
    images: ImageConfig
    segmentation: SegmentationConfig
    plant_detection: PlantDetetionConfig
    alignment: AlignmentConfig
    weed_filter: WeedFilterConfig
    plant_grouping: PlantGroupingConfig
    dataframes: DataframesConfig
    dataset_generation: DatasetGenerationConfig
    quality_evaluation: QualityEvaluationConfig
    orthoimages: OrthoimagesConfig
    plot: PlotConfig
