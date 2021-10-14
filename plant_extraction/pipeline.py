import os, glob

from fluidml import Flow, Swarm
from fluidml.common import Task, Resource
from fluidml.flow import GridTaskSpec, TaskSpec
from fluidml.storage import LocalFileStore
from fluidml.common.logging import configure_logging
from fluidml.visualization import visualize_graph_interactive

from plant_extraction.tasks import *
from plant_extraction.config import PipelineConfig

import yaml
from dacite import from_dict

class Pipeline():

    def __init__(self, config_file_path: str):
        with open(config_file_path, "r") as file:
            self.config = from_dict(data_class = PipelineConfig, data = yaml.safe_load(file))
    
    def build(self):

        self.tasks = []

        for field_id in self.config.images.field_ids:

            segment_soil_plants_kwargs = {
                "image_path": sorted(glob(os.path.join(self.config.images.source_dir, f"{field_id}_*{self.config.images.img_format}"))),
                "image_channels": [self.config.images.channels],
                "use_watershed": self.config.segmentation.use_watershed,
                "veg_index": self.config.segmentation.veg_index,
                "max_coverratio": self.config.plant_detection.max_detection_coverratio,
                "make_orthoimage": self.config.orthoimages.make_orthoimages,
                "orthoimage_dir": self.config.orthoimages.save_dir,
                "plot_result": self.config.plot.plant_detection.make_plots,
                "plot_dir": self.config.plot.plant_detection.save_dir,
                "plot_format": self.config.plot.plant_detection.img_format,
                "plot_dpi": self.config.plot.plant_detection.dpi,
                "plot_cmap": self.config.plot.plant_detection.cmap
                }
            segment_soil_plants_task = GridTaskSpec(
                task=SegmentSoilPlants,
                name=field_id+"_SegmentSoilPlants",
                gs_config=segment_soil_plants_kwargs,
                gs_expansion_method = "product",
                publishes=[
                    "cover_ratio",
                    "field_id",
                    "date",
                    "px_resolution",
                    "segmentation_mask",
                    "raster_meta"
                    ]
                )

            fit_grow_function_kwargs = {
                "plot_result": self.config.plot.plant_detection.make_plots,
                "plot_dir": self.config.plot.plant_detection.save_dir,
                "plot_format": self.config.plot.plant_detection.img_format,
                "plot_dpi": self.config.plot.plant_detection.dpi
                }
            fit_grow_function_task = TaskSpec(
                task=FitGrowFunction,
                name=field_id+"_FitGrowFunction",
                config=fit_grow_function_kwargs,
                publishes=[
                    "grow_function_fit_params",
                    "grow_function_cov_matrix"#,
                    #"grow_function_cover_ratios",
                    #"dates"
                    ],
                reduce=True
                )

            extract_plant_positions_kwargs = {
                "min_peak_distance": self.config.plant_detection.min_peak_distance,
                "peak_threshold": self.config.plant_detection.peak_threshold,
                "gauss_sigma_bounds": self.config.plant_detection.gauss_sigma_bounds,
                "use_growfunction": self.config.segmentation.use_growfunction_coverratio,
                "make_orthoimage": self.config.orthoimages.make_orthoimages,
                "orthoimage_dir": self.config.orthoimages.save_dir,
                "plot_result": self.config.plot.plant_detection.make_plots,
                "plot_dir": self.config.plot.plant_detection.save_dir,
                "plot_format": self.config.plot.plant_detection.img_format,
                "plot_dpi": self.config.plot.plant_detection.dpi,
                "plot_cmap": self.config.plot.plant_detection.cmap
                }   
            extract_plant_positions_task = TaskSpec(
                task=ExtractPlantPositions,
                name=field_id+"_ExtractPlantPositions",
                config=extract_plant_positions_kwargs,
                publishes=[
                    "plant_positions",
                    "utm_transform"
                    ]
                )

            load_peaks_kwargs = {
                "field_id": field_id,
                "plot_result": self.config.plot.plant_grouping.make_plots,
                "plot_dir": self.config.plot.plant_grouping.save_dir,
                "plot_format": self.config.plot.plant_grouping.img_format,
                "plot_dpi": self.config.plot.plant_grouping.dpi,
                "plot_cmap": self.config.plot.plant_grouping.cmap
                }
            load_peaks_task = TaskSpec(
                task=LoadPeaks,
                name=field_id+"_LoadPeaks",
                config=load_peaks_kwargs,
                publishes=[
                    "point_cloud",
                    "field_id",
                    "layers",
                    "cover_ratios",
                    "printdates",
                    "dates",
                    "gps_transforms",
                    "px_resolutions",
                    "utm_transforms",
                    "segmentation_masks"
                    ],
                reduce=True
                )

            align_points_kwargs = {
                "max_centroid_distance_cpd": self.config.alignment.max_centroid_distance,
                "max_centroid_distance_group": self.config.plant_grouping.max_centroid_distance,
                "make_orthoimage": self.config.orthoimages.make_orthoimages,
                "orthoimage_dir": self.config.orthoimages.save_dir,
                "plot_result": self.config.plot.plant_grouping.make_plots,
                "plot_dir": self.config.plot.plant_grouping.save_dir,
                "plot_format": self.config.plot.plant_grouping.img_format,
                "plot_dpi": self.config.plot.plant_grouping.dpi,
                "plot_cmap": self.config.plot.plant_grouping.cmap
                }
            align_points_task = TaskSpec(
                task=AlignPoints,
                name=field_id+"_AlignPoints",
                config=align_points_kwargs,
                publishes=[
                    "point_cloud_aligned",
                    "point_cloud_mean",
                    "align_transform"
                    ]
                )
            
            align_croplines_kwargs = {
                "plot_result": self.config.plot.plant_grouping.make_plots,
                "plot_dir": self.config.plot.plant_grouping.save_dir,
                "plot_format": self.config.plot.plant_grouping.img_format,
                "plot_dpi": self.config.plot.plant_grouping.dpi,
                "plot_cmap": self.config.plot.plant_grouping.cmap
                }
            align_croplines_task = TaskSpec(
                task=AlignCroplines,
                name=field_id+"_AlignCroplines",
                config=align_croplines_kwargs,
                publishes=[
                    "point_cloud_rotated",
                    "rotation_angle",
                    "median_cropline_distance"
                    ]
                )

            find_croplines_kwargs = {
                "plot_result": self.config.plot.plant_grouping.make_plots,
                "plot_dir": self.config.plot.plant_grouping.save_dir,
                "plot_format": self.config.plot.plant_grouping.img_format,
                "plot_dpi": self.config.plot.plant_grouping.dpi
                }
            find_croplines_task = TaskSpec(
                task=FindCroplines,
                name=field_id+"_FindCroplines",
                config=find_croplines_kwargs,
                publishes=[
                    "croplines_ypos"
                    ]
                )

            filter_weed_kwargs = {
                "threshold_factor": self.config.weed_filter.threshold_factor,
                "plot_result": self.config.plot.plant_grouping.make_plots,
                "plot_dir": self.config.plot.plant_grouping.save_dir,
                "plot_format": self.config.plot.plant_grouping.img_format,
                "plot_dpi": self.config.plot.plant_grouping.dpi
                }
            filter_weed_task = TaskSpec(
                task=FilterWeed,
                name=field_id+"_FilterWeed",
                config=filter_weed_kwargs,
                publishes=[
                    "point_cloud_weedfiltered",
                    "point_cloud_aligned_weedfiltered",
                    "point_cloud_rotated_weedfiltered",
                    "layers_weedfiltered"
                    ]
                )

            group_points_kwargs = {
                "max_centroid_distance": self.config.plant_grouping.max_centroid_distance
                }
            group_points_task = TaskSpec(
                task=GroupPoints,
                name=field_id+"_GroupPoints",
                config=group_points_kwargs,
                publishes=[
                    "point_cloud_weedfiltered_grouped",
                    "point_cloud_aligned_weedfiltered_grouped",
                    "point_cloud_rotated_weedfiltered_grouped",
                    "group_labels",
                    "layers_weedfiltered_grouped"
                    ]
                )

            sort_group_labels_kwargs = {
                "plot_result": self.config.plot.plant_grouping.make_plots,
                "plot_dir": self.config.plot.plant_grouping.save_dir,
                "plot_format": self.config.plot.plant_grouping.img_format,
                "plot_dpi": self.config.plot.plant_grouping.dpi,
                "plot_cmap": self.config.plot.plant_grouping.cmap
                }
            sort_group_labels_task = TaskSpec(
                task=SortGroupLabels,
                name=field_id+"_SortGroupLabels",
                config=sort_group_labels_kwargs,
                publishes=[
                    "group_labels_sorted",
                    "group_cropline_ids_sorted",
                    "group_sizes_sorted"
                    ]
                )

            save_plants_dataframe_kwargs = {
                "save_dir": self.config.dataframes.save_dir
                }
            save_plants_dataframe_task = TaskSpec(
                task=SavePlantsDataFrame,
                name=field_id+"_SavePlantsDataFrame",
                config=save_plants_dataframe_kwargs,
                publishes=[
                    "_dummy"
                    ]
                )
    
            export_plant_positions_kwargs = {
                "df_dir": self.config.dataframes.save_dir
                }
            export_plant_positions_task = TaskSpec(
                task=ExportPlantPositions,
                name=field_id+"_ExportPlantPositions",
                config=export_plant_positions_kwargs,
                publishes=[
                    "_dummy"
                    ],
                )
            
            # task dependencies
            fit_grow_function_task.requires(segment_soil_plants_task)
            extract_plant_positions_task.requires(segment_soil_plants_task)
            load_peaks_task.requires([segment_soil_plants_task, extract_plant_positions_task])
            align_points_task.requires(load_peaks_task)
            align_croplines_task.requires([align_points_task, load_peaks_task])
            find_croplines_task.requires([align_croplines_task, align_points_task, load_peaks_task])
            filter_weed_task.requires([find_croplines_task, align_points_task, align_croplines_task, load_peaks_task])
            group_points_task.requires(filter_weed_task)
            sort_group_labels_task.requires([group_points_task, find_croplines_task, load_peaks_task])
            save_plants_dataframe_task.requires([align_points_task, align_croplines_task,
                                                find_croplines_task, group_points_task, sort_group_labels_task, load_peaks_task])
            export_plant_positions_task.requires([save_plants_dataframe_task, load_peaks_task])

            # add to list
            self.tasks.extend([
                segment_soil_plants_task,
                fit_grow_function_task,
                extract_plant_positions_task,
                load_peaks_task,
                align_points_task,
                align_croplines_task,
                find_croplines_task,
                filter_weed_task,
                group_points_task,
                sort_group_labels_task,
                save_plants_dataframe_task,
                export_plant_positions_task  
            ])

            # optional tasks
            if self.config.quality_evaluation.ground_truth_data_dir:
                evaluate_detection_quality_kwargs = {
                    "df_dir": self.config.dataframes.save_dir,
                    "image_dir": self.config.images.source_dir,
                    "ground_truth_dir": self.config.quality_evaluation.ground_truth_data_dir,
                    "image_channels": self.config.images.channels,
                    "max_distance": self.config.quality_evaluation.max_detection_distance,
                    "save_dir": self.config.dataframes.save_dir,
                    "plot_result": self.config.plot.quality_evaluation.make_plots,
                    "plot_dir": self.config.plot.quality_evaluation.save_dir,
                    "plot_format": self.config.plot.quality_evaluation.img_format,
                    "plot_dpi": self.config.plot.quality_evaluation.dpi
                    }
                evaluate_detection_quality_task = TaskSpec(
                    task=EvaluateDetectionQuality,
                    name=field_id+"_EvaluateDetectionQuality",
                    config=evaluate_detection_quality_kwargs,
                    publishes=[
                        "detection_quality",
                        "direct_detections_xy",
                        "indirect_detections_xy",
                        "ground_truth_xy"
                        ]
                    )

                evaluate_detection_quality_task.requires([load_peaks_task, save_plants_dataframe_task])
                self.tasks.append(evaluate_detection_quality_task)
        
            if self.config.dataset_generation.shape:
                make_image_dataset_kwargs = {
                    "df_dir": self.config.dataframes.save_dir,
                    "source_tiff_dir": self.config.images.source_dir,
                    "source_channels": self.config.images.channels,
                    "export_channels": self.config.dataset_generation.channels,
                    "export_shape": self.config.dataset_generation.shape,
                    "ann_df_path": self.config.dataset_generation.annotations.df_path,
                    "ann_gps_name": self.config.dataset_generation.annotations.gps_column,
                    "ann_values_name": self.config.dataset_generation.annotations.values_column,
                    "tol_distance": self.config.dataset_generation.annotations.tol_distance,
                    "save_dir": self.config.dataset_generation.save_dir
                    }
                make_image_dataset_task = TaskSpec(
                    task=MakeImageDataset,
                    name=field_id+"_MakeImageDataset",
                    config=make_image_dataset_kwargs,
                    publishes=[
                        "field_id"
                        ]
                    )
                make_image_dataset_task.requires([load_peaks_task, save_plants_dataframe_task])
                self.tasks.append(make_image_dataset_task)

        last_tasks = self.tasks[len(self.tasks)//len(self.config.images.field_ids)-1::len(self.tasks)//len(self.config.images.field_ids)]

        merge_plants_dataframe_kwargs = {
            "save_dir": self.config.dataframes.save_dir
            }
        merge_plants_dataframe_task = TaskSpec(
            task=MergePlantsDataFrame,
            name="MergePlantsDataFrame",
            config=merge_plants_dataframe_kwargs,
            publishes=[
                "_dummy"
                ],
            reduce=True
            )

        merge_plants_dataframe_task.requires(last_tasks)
        self.tasks.append(merge_plants_dataframe_task)

        if self.config.dataset_generation.shape:
            merge_bounding_box_files_kwargs = {
                "img_root_dir": self.config.dataset_generation.save_dir,
                }
            merge_bounding_box_files_task = TaskSpec(
                task=MergeBoundingBoxFiles,
                name="MergeBoundingBoxFiles",
                config=merge_bounding_box_files_kwargs,
                publishes=[
                    "_dummy"
                    ],
                reduce=True
                )
            merge_bounding_box_files_task.requires(last_tasks)
            self.tasks.append(merge_bounding_box_files_task)


    def run(self):
        
        results_store = LocalFileStore(base_dir=os.path.join(os.path.abspath(''), self.config.fluidml.save_dir))
    
        configure_logging(level=self.config.fluidml.logging_level)
            
        with Swarm(
            n_dolphins=self.config.fluidml.n_workers,
            results_store=results_store) as swarm:

            flow = Flow(swarm=swarm)
            flow.create(task_specs=self.tasks)
            
            # optional graph visualization
            if self.config.fluidml.visualize_graph:
                visualize_graph_interactive(
                    graph=flow.task_spec_graph)

            results = flow.run(force=self.config.fluidml.force)
