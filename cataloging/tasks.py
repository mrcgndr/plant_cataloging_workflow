import datetime as dt
import gc
import json
import logging
import os
import pickle
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

import h5py
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
import simplekml
from cataloging.vi import gliImage, ngrdiImage, osaviImage
from fluidml.common import Task
#from PIL import Image
from pycpd import RigidRegistration
from pykml import parser
from rasterio.enums import Resampling
from rasterio.transform import rowcol, xy
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
#from skimage.exposure import equalize_adapthist
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.transform import hough_line, hough_line_peaks, resize
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

# suppress pickle 'error' from rasterio
logging.Logger.manager.loggerDict['rasterio'].setLevel(logging.CRITICAL)
logging.Logger.manager.loggerDict['matplotlib'].setLevel(logging.CRITICAL)

import warnings

warnings.filterwarnings("ignore")

mpl.use('Agg')

def read_raster(
        image_path: str,
        all_channels: np.array,
        channels: List[str]
    ):
    ch = [np.argmax(all_channels == c)+1 for c in channels]
    raster = rio.open(image_path)
    if raster.dtypes[0] == "float32":
        data = raster.read(ch, fill_value=np.nan)
        data /= np.nanmax(data)
    elif raster.dtypes[0] == "uint8":
        if "alpha" in all_channels:
            data = raster.read(ch).astype(np.float32)
            alpha_ch = raster.read(int(np.argmax(all_channels == "alpha")+1))
            for d in data[:,:]:
                d[alpha_ch == 0] = np.nan
        else:
            data = raster.read(ch, fill_value=0).astype(np.float32)
    else:
        raise NotImplementedError()
    return np.transpose(data, axes=(1,2,0))

def write_onechannel_raster(
        image_path: str,
        image: np.array,
        meta: Dict, dtype: str
    ):
    if dtype == 'float32':
        meta.update({
                    'dtype': 'float32',
                    'height': image.shape[0],'count': 1,'nodata': -32767,
                    'width': image.shape[1]})
    elif dtype == 'uint8':
        meta.update({
                    'dtype': 'uint8',
                    'height': image.shape[0],'count': 1,'nodata': 0,
                    'width': image.shape[1]})
    else:
        raise NotImplementedError()
    with rio.open(image_path, "w", **meta) as dest:
        dest.write(image,1)

def calc_m_per_px(
        raster_meta: Dict
    ) -> float:
    # read CRS of rasterio data
    proj_crs = pyproj.crs.CRS.from_user_input(raster_meta["crs"])
    # GPS coordinates of anchor point
    lon0, lat0 = xy(raster_meta["transform"],0,0)
    # calculate UTM zone
    utm_zone = int(np.floor((lon0/360)*60+31))

    utm = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84')
    UTM0_x, UTM0_y = utm(*xy(raster_meta["transform"],0,0))
    UTM1_x, UTM1_y = utm(*xy(raster_meta["transform"],0,1)) 
    UTM2_x, UTM2_y = utm(*xy(raster_meta["transform"],1,0))

    # calculate unit pixel distances
    pxx = abs(UTM1_x - UTM0_x)
    pxy = abs(UTM2_y - UTM0_y)
    # take mean (assume quadratic pixels)
    m_per_px = np.mean([pxx, pxy])
    return m_per_px

def px_to_utm(
        point_cloud: np.ndarray,
        raster_meta: Dict
    ) -> Tuple[np.ndarray, pyproj.proj.Proj]:
    # read CRS of rasterio data
    proj_crs = pyproj.crs.CRS.from_user_input(raster_meta["crs"])
    # GPS coordinates of point cloud
    lon, lat = np.asarray(xy(raster_meta["transform"],*point_cloud.T))
    # calculate UTM zone
    utm_zone = int(np.floor((lon.mean()/360)*60+31))

    utm_transform = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84')
    utm = np.asarray(utm_transform(lon, lat)).T

    return utm, utm_transform

def readCoordsFromKml(
        filename: str
    ) -> np.ndarray:
    with open(filename, "r") as kmlfile:
        root = parser.parse(kmlfile).getroot()
        lonlat = []
        for c in root.Document.iterchildren():
            lonlat.append([float(x) for x in c.Point.coordinates.text.split(",")[:2]])
        lonlat = np.asarray(lonlat)    
    
    return lonlat

def growFunction(
        x: float,
        g: float,
        lg: float,
        xg: float,
        d: float,
        ld: float,
        xd: float
    ) -> float:
    if d > 0:
        return (g/(1+np.exp(-lg*(x-xg)))) - d/(1+np.exp(-ld*(x-xd)))
    else:
        return (g/(1+np.exp(-lg*(x-xg))))

def cumDays(
        observation_dates: Union[List[float],np.array]
    ) -> np.array:
    cum_days = np.cumsum([d.days for d in np.diff(np.sort(observation_dates))]).astype(float)
    cum_days = np.hstack((0, cum_days))
    return cum_days

def growScaling(
        cum_days: np.array,
        bounds: Tuple,
        grow_func_params: np.array
    ) -> np.array:
    earliest, latest = bounds
    grow_func = growFunction(cum_days, *grow_func_params)
    maxgrow_val = np.max(grow_func)
    grow_func = (grow_func - grow_func[0]) / (maxgrow_val - grow_func[0])
    scaled = grow_func * (latest - earliest) + earliest
    return scaled

def makeDirectory(
        directory: str
    ) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

def group_points(
        points: np.array,
        layers: np.array,
        max_dist: float
    ) -> Tuple[np.array, np.array]:
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)

    # initialization
    # -> all labels to -1
    labels = -np.ones_like(layers)
    # all given layers
    uni_layers = np.unique(layers)
    # -> give points of first layer individual group labels
    labels[layers == uni_layers[0]] = np.arange(np.sum(layers == uni_layers[0]))
    # -> first evaluation point cloud: first layer
    centroids = points[layers == uni_layers[0]]

    ind = np.arange(len(points))
    
    for i in range(1, len(uni_layers)):
        # fit nearest neighbor model
        nn.fit(centroids)

        # evaluate on next layer
        dist, ass_group = nn.kneighbors(points[layers == uni_layers[i]])
        dist = dist.flatten()
        ass_group = ass_group.flatten()

        # exclude points that have more than max_dist distance to a neighbor
        # new_member array:
        #  1 = valid member candidate for existing group
        #  0 = valid member candidate for new group
        # -1 = excluded due to multiple candidates for a single group
        new_member = (dist <= max_dist).astype(int)
        # if multiple (valid!) points are assigned to the same group, take the nearest
        valid = np.copy(new_member).astype(bool)
        valid_ind = np.arange(len(valid))[valid]
        for j, counts in enumerate(np.bincount(ass_group[valid])):
            if counts > 1:
                ass_group_ind = valid_ind[ass_group[valid] == j]
                best_ind = ass_group_ind[np.argsort(dist[ass_group_ind])]
                new_member[best_ind[1:]] = -1

        # assign the group labels to the new members
        layer_ind = ind[layers == uni_layers[i]]
        old_layer_ind = layer_ind[new_member == 1]
        labels[old_layer_ind] = ass_group[new_member == 1]

        # give new group labels to points not registered so far
        new_layer_ind = layer_ind[new_member == 0] 
        labels[new_layer_ind] = np.arange(labels.max()+1, labels.max()+1+len(new_layer_ind))

        # new reference cloud are the centroids of the so far accumulated clusters
        centroids = np.stack([np.mean(points[labels == label], axis=0) for label in range(labels.max()+1)]) 
        
    return labels, centroids

def inverse_transform(
        xy_centered_aligned,
        xy_center,
        transform_coeffs
    ):
    s   = transform_coeffs[0]
    rot = np.deg2rad(transform_coeffs[1])
    t   = transform_coeffs[2:]

    rot_inv = np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]])

    return rot_inv@(xy_centered_aligned-t).T/s + xy_center

def add_non_detected(
        df_less: pd.DataFrame,
        df_meta: pd.DataFrame
    ) -> pd.DataFrame:
    dates = np.unique(df_meta["date"])
    xy_center = df_meta["xy_center"].iloc[0]
    df_add = pd.DataFrame()
    for g_id in np.unique(df_less["group_id"]):
        df_group = df_less[df_less["group_id"] == g_id]
        missing_dates = dates[np.isin(dates, df_group["date"], invert=True)]
        for d in missing_dates:
            xy_centered_aligned = df_group["xy_centered_aligned_cm"].mean(axis=0) # group centroid [cm (UTM)]
            cropline_y = df_group["y_cropline_rotated_cm"].iloc[0]
            align_transform = df_meta[df_meta["date"] == d]["align_transform"].iloc[0]
            gps_transform = df_meta[df_meta["date"] == d]["gps_transform"].iloc[0]
            utm_transform = df_meta[df_meta["date"] == d]["utm_transform"].iloc[0]
            #cr =  df_meta[df_meta["date"] == d]["cover_ratio"].values
            #mc =  df_meta[df_meta["date"] == d]["align_median_confidence"].values
            xy_backtrans = inverse_transform(xy_centered_aligned, xy_center, align_transform)
            lonlat_backtrans = utm_transform(*xy_backtrans/100., inverse=True)
            df_add = df_add.append(
                dict([("field_id"                        , df_group["field_id"].iloc[0]),
                      ("date"                            , d),
                      ("group_id"                        , g_id),
                      ("group_size"                      , df_group["group_size"].iloc[0]),
                      ("group_cropline_id"               , df_group["group_cropline_id"].iloc[0]),
                      ("xy_cm"                           , xy_backtrans),
                      ("xy_px"                           , list(rowcol(gps_transform, *lonlat_backtrans))),
                      ("lonlat"                          , lonlat_backtrans),
                      ("xy_centered_aligned_cm"          , xy_centered_aligned),
                      ("xy_centroid_centered_aligned_cm" , xy_centered_aligned),
                      ("y_cropline_rotated_cm"           , cropline_y),
                      ("centroid_dist_cm"                , 0.),
                      ("detected"                        , False)]), ignore_index=True)
        
    return df_add

def filterGoodPlantsByPercDet(
        plants_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        filter_coverratio: float,
        perc_min_det: float
    ) -> pd.DataFrame:
    
    plants_meta_df = plants_df.merge(meta_df, on=["date", "field_id"], how="left")
    
    n_dates = len(np.unique(meta_df["date"]))
    
    # good plant group := at least perc_min_det direct detection ratio up to certain given cover ratio
    good_idx = []    
    for f_id in np.unique(meta_df["field_id"]):
        n_counts_below_cr_thres = np.sum(np.unique(plants_meta_df[plants_meta_df["field_id"]==f_id]["cover_ratio"]) <= filter_coverratio)
        groups, counts = np.unique(plants_meta_df[(plants_meta_df["field_id"]==f_id) & (plants_meta_df["cover_ratio"] <= filter_coverratio) & (plants_meta_df["detected"] == True)]["group_id"], return_counts=True)
        interest_groups = groups[counts/float(n_counts_below_cr_thres) >= perc_min_det]
        candidates = plants_meta_df[(plants_meta_df["field_id"]==f_id) & (np.isin(plants_meta_df["group_id"], interest_groups))]
        for g_id in interest_groups:
            cand_group = candidates[candidates["group_id"]==g_id]
            if len(cand_group)==n_dates:
                good_idx.extend(cand_group.index)
    good_df = plants_meta_df.loc[good_idx].sort_values(["field_id", "group_id", "date"])
    
    return good_df


class SegmentSoilPlants(Task):
    
    def __init__(
            self,
            image_path: str,
            image_channels: List[str],
            veg_index: str,
            use_watershed: bool,
            max_coverratio: float,
            make_orthoimage: bool,
            orthoimage_dir: str,
            plot_result: bool,
            plot_dir: str,
            plot_format: str,
            plot_dpi: int,
            plot_cmap: str
        ):
        super().__init__()
        self.image_path = image_path
        self.image_channels = np.asarray(image_channels)
        self.veg_index = veg_index
        self.use_watershed = use_watershed
        self.max_coverratio = max_coverratio
        self.make_orthoimage = make_orthoimage
        self.orthoimage_dir = orthoimage_dir
        self.plot_result = plot_result
        self.plot_dir    = plot_dir
        self.plot_format = plot_format
        self.plot_dpi    = plot_dpi
        self.plot_cmap   = plot_cmap
           
    def plot_raw(
            self
        ):
        logger.info(f"{self.name}-{self.date.date()} -> Plot raw image.")
        if len(self.image_channels) < 4:
            n_rows, n_cols = 1, len(self.image_channels)
        else:
            n_rows, n_cols = 2, len(self.image_channels)//2
        fig, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(self.width/500*n_cols, self.height/800*n_rows))
        data = read_raster(self.image_path, self.image_channels, self.image_channels)
        for (i, (a, c)) in enumerate(zip(ax.ravel(), self.image_channels)):
            im = a.imshow(data[:,:,i], cmap=self.plot_cmap)
            try:
                fig.colorbar(im, ax=a)
            except:
                pass
            a.set(xlabel='x', ylabel='y', title = c, aspect='equal')
        fig.suptitle("raw image data")
        fig.savefig(os.path.join(self.plot_dir, f"{self.field_id}_{self.date.date()}_01_channels"+self.plot_format), dpi=self.plot_dpi, bbox_inches='tight')
        fig.clf()
        del data, fig, ax, im
        plt.close("all")
        gc.collect()
    
    def plot_segmentation(
            self
        ):
        logger.info(f"{self.name}-{self.date.date()} -> Plot segmentation image.")
        fig = plt.figure(figsize=(3*self.width/500, self.height/500), tight_layout=True)
        gridspec = gs.GridSpec(1,3,width_ratios=[2,1,2], figure=fig)
        ax1 = fig.add_subplot(gridspec[0])
        ax2 = fig.add_subplot(gridspec[1])
        ax3 = fig.add_subplot(gridspec[2])

        m = ax1.imshow(self.vi_image.astype(float), cmap=self.plot_cmap, vmin=-1, vmax=1)
        cb = fig.colorbar(m, ax=ax1)
        cb.set_label("VI")
        ax1.set(title=f"{self.veg_index} image", xlabel="px", ylabel="px")
        ax2.hist(self.vi_image[np.isfinite(self.vi_image)], bins=256, orientation="horizontal", color="C0")
        ax2.set(title=f"{self.veg_index} value distribution", ylim=(-1,1), xlabel="counts", xscale="log")
        if self.cover_ratio_est < 0.01:
            ax2.axhline(self.thres, c='r', label=f"Threshold (99-percentile): {self.thres:.2f}")
        else:
            ax2.axhline(self.thres, c='r', label=f"Threshold (Otsu): {self.thres:.2f}")
        ax2.legend()
        ax3.imshow(self.seg_mask, cmap=self.plot_cmap)
        ax3.set(title=f"Segmented plant area (cover ratio: {100.*self.cover_ratio:.2f} %)", xlabel="px", ylabel="px")  
        fig.savefig(os.path.join(self.plot_dir, f"{self.field_id}_{self.date.date()}_02_segmentation"+self.plot_format), dpi=self.plot_dpi, bbox_inches='tight')
        fig.clf()
        plt.close("all")
        del fig, ax1, ax2, ax3
        gc.collect()
        
    def run(
            self
        ):
        try:
            self.field_id, d = os.path.basename(self.image_path).replace(".tif", "").split("_")[:2]
            year  = int(d[:4])
            month = int(d[4:6])
            day   = int(d[6:8])
            self.date = dt.datetime(year, month, day)
        except:
            logger.error(f"Wrong image path or no files found: {self.image_path}")

        logger.info(f"{self.name}-{self.date.date()} -> Load image.")
        raster = rio.open(self.image_path)
        raster_meta = raster.meta
        self.height, self.width = raster.shape
        px_res = calc_m_per_px(raster_meta)*100. # cm/px
        logger.info(f"{self.name}-{self.date.date()} -> Calculated resolution: {px_res:.4f} cm/px.")
        del raster
        gc.collect()
        
        # calculate Vegetation Index which has values in [-1,1]
        if self.veg_index == "NGRDI":
            channels = read_raster(self.image_path, self.image_channels, ["R", "G"])
            self.vi_image = ngrdiImage(R = channels[:,:,0], G = channels[:,:,1])
            est_thres = 0
        elif self.veg_index == "GLI":
            channels = read_raster(self.image_path, self.image_channels, ["R", "G", "B"])
            self.vi_image = gliImage(R = channels[:,:,0], G = channels[:,:,1], B = channels[:,:,2])
            est_thres = 0.2
        elif self.veg_index == "OSAVI":
            channels = read_raster(self.image_path, self.image_channels, ["R", "NIR"])
            self.vi_image = osaviImage(R = channels[:,:,0], NIR = channels[:,:,1], y_osavi = 0.6)
            est_thres = 0.25
        del channels
        gc.collect()

        # cover ratio estimation
        self.cover_ratio_est = np.nansum(self.vi_image >= est_thres)/np.sum(np.isfinite(self.vi_image))
        logger.info(f"{self.name}-{self.date.date()} -> Use {self.veg_index} Vegetation Index. Cover ratio estimation: {self.cover_ratio_est*100.:.2f} %")
        
        if self.cover_ratio_est <= self.max_coverratio:

            # calculate threshold with Otsu's method
            if self.cover_ratio_est < 0.01:
                self.thres = np.percentile(self.vi_image[np.isfinite(self.vi_image)], 99)
                logger.warn(f"{self.name}-{self.date.date()} -> Estimated cover ratio below 1 % -> Take 99-percentile as threshold: {self.thres:.2f}")
            else:
                self.thres = threshold_otsu(self.vi_image[np.isfinite(self.vi_image)])
                logger.info(f"{self.name}-{self.date.date()} -> Otsu threshold: {self.thres:.2f}")

            # segmentation 
            if self.use_watershed:
                logger.info(f"{self.name}-{self.date.date()} -> Segment soil and plants with watershed method.")
                markers = np.zeros_like(self.vi_image, dtype=np.uint8)
                markers[self.vi_image <= self.thres] = 1 # soil
                markers[self.vi_image  > self.thres] = 2 # plant
                self.seg_mask = (watershed(self.vi_image, markers) - 1).astype(bool) # True -> plant, False -> soil
                del markers
            else:
                logger.info(f"{self.name}-{self.date.date()} -> Segment soil and plants without watershed method.")
                self.seg_mask = np.zeros_like(self.vi_image, dtype=bool) # True -> plant, False -> soil
                self.seg_mask[self.vi_image  > self.thres] = True # plant
            
            self.cover_ratio = np.sum(self.seg_mask)/np.sum(np.isfinite(self.vi_image))
            
            logger.info(f"{self.name}-{self.date.date()} -> Cover ratio recalculated: {self.cover_ratio*100.:.2f} %")

            if self.plot_result:
                makeDirectory(self.plot_dir)
                self.plot_segmentation() 
                gc.collect()

        else:
            logger.warn(f"{self.name}-{self.date.date()} -> Estimated cover ratio ({self.cover_ratio_est*100.:.2f} %) is too high to extract plants -> Skip plot.")
            self.seg_mask = []
            self.cover_ratio = self.cover_ratio_est

        self.save(obj=self.seg_mask, name="segmentation_mask", type_='pickle')
        self.save(obj=self.cover_ratio, name="cover_ratio", type_='json')
        self.save(obj=self.field_id, name="field_id", type_='json')
        self.save(obj=self.date, name="date", type_='pickle')
        self.save(obj=raster_meta, name="raster_meta", type_='pickle')
        self.save(obj=px_res, name="px_resolution", type_='json')

        if (self.make_orthoimage) and (self.seg_mask != []):
            makeDirectory(self.orthoimage_dir)
            logger.info(f"{self.name}-{self.date.date()} -> Save segmentation mask as orthoimage.")
            write_onechannel_raster(os.path.join(self.orthoimage_dir, f"{self.field_id}_{self.date.date()}_segmentation.tif"),
                                    np.uint8(self.seg_mask*255),
                                    raster_meta,
                                    "uint8")

        # plot raw channel information
        if self.plot_result:
            makeDirectory(self.plot_dir)
            self.plot_raw()
            gc.collect()

                      
class FitGrowFunction(Task):
    
    def __init__(
            self,
            plot_result: bool,
            plot_dir: str,
            plot_format: str,
            plot_dpi: int
        ):
        super().__init__()
        self.plot_result = plot_result
        self.plot_dir    = plot_dir
        self.plot_format = plot_format
        self.plot_dpi    = plot_dpi

    def plot(
            self
        ):
        logger.info(f"{self.name} -> Plot Grow function.")
        g, lg, xg, d, ld, xd = self.fit
        cd = np.linspace(0, self.cum_days[-1], 1000)
        cal_days = [self.observation_dates[0] + dt.timedelta(days=x) for x in self.cum_days]
        
        fig, ax = plt.subplots()
        ax.scatter(self.cum_days, self.cover_ratios, label="observations")
        if d > 0:
            label = r"grow function fit: $f(x)=\frac{g}{1+e^{-\lambda_g(x-x_g)}}-\frac{d}{1+e^{-\lambda_d(x-x_d)}}$"+f"\n$g$={g:.4g}, $\\lambda_g$={lg:.4g}, $x_g$={xg:.4g}\n$d$={d:.4g}, $\\lambda_d$={ld:.4g}, $x_d$={xd:.4g}"
        else:
            label = r"grow function fit: $f(x)=\frac{g}{1+e^{-\lambda_g(x-x_g)}}$"+f"\n$g$={g:.4g}, $\\lambda_g$={lg:.4g}, $x_g$={xg:.4g}"
        ax.plot(cd, growFunction(cd, *self.fit), c="r", label=label)
        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
        ax.set(xlabel="days", ylabel="cover ratio")
        ax.legend()
        ax.grid()
        ax_dt = ax.twiny()
        ax_dt.set_xlim(map(lambda cd: self.observation_dates[0] + dt.timedelta(days=cd), ax.get_xlim()))
        ax_dt.set_xlabel("calendar date")
        ax_dt.set_xticks(cal_days)
        ax_dt.tick_params(axis='x', labelrotation=90)
        ax.set(title=f"{self.field_id}: grow function fit") 
        savename = os.path.join(self.plot_dir, f"{self.field_id}_grow_function"+self.plot_format)
        fig.savefig(savename, dpi=self.plot_dpi, bbox_inches='tight') 
        plt.close("all")
        del fig, ax, ax_dt
                
    def run(
            self,
            reduced_results: List[Dict[str, Dict]]
        ):        
        cover_ratios = []
        observation_dates = []
        
        for r in reduced_results:
            cover_ratios.append(r["result"]["cover_ratio"])
            observation_dates.append(r["result"]["date"])
        
        observation_dates = np.asarray(observation_dates)
        cover_ratios = np.asarray(cover_ratios)
        
        sort = np.argsort(observation_dates)
        self.observation_dates = observation_dates[sort]
        self.cover_ratios = cover_ratios[sort]
        
        self.cum_days = cumDays(self.observation_dates)

        self.field_id = reduced_results[0]["result"]["field_id"]
        
        try:
            self.fit, self.cov = curve_fit(growFunction, self.cum_days, self.cover_ratios,
                                        p0=[0.8, 0.1, self.cum_days[-1]/3, 0.3, 0.1, 2*self.cum_days[-1]/3],
                                        maxfev=1000000)

            # calculate corrected cover ratios with grow function
            #gf_cover_ratio = growFunction(self.cum_days, *self.fit)
            
            
            #self.save(obj=gf_cover_ratio, name="grow_function_cover_ratios", type_='pickle')
            #self.save(obj=self.observation_dates, name="dates", type_='pickle')
            logger.info(f"{self.name} -> Grow function fitted")

            if self.plot_result:
                makeDirectory(self.plot_dir)
                self.plot()
                gc.collect()  

        except Exception as e:
            self.fit = np.nan
            self.cov = np.nan
            logger.warning(f"{self.name} -> Grow function could not be fitted. Error: {e}")
        
        self.save(obj=self.fit, name="grow_function_fit_params", type_='pickle')
        self.save(obj=self.cov, name="grow_function_cov_matrix", type_='pickle')
            
            
class ExtractPlantPositions(Task):
    
    def __init__(
            self,
            min_peak_distance: float,
            peak_threshold: float,
            gauss_sigma_bounds: Tuple[float, float],
            use_growfunction: bool,
            make_orthoimage: bool,
            orthoimage_dir: str,
            plot_result: bool,
            plot_dir: str,
            plot_format: str,
            plot_dpi: int,
            plot_cmap: str
        ):
        super().__init__()
        self.min_peak_distance = min_peak_distance
        self.peak_threshold = peak_threshold
        self.gauss_sigma_bounds = gauss_sigma_bounds
        self.use_growfunction = use_growfunction
        self.make_orthoimage = make_orthoimage
        self.orthoimage_dir = orthoimage_dir
        self.plot_result = plot_result
        self.plot_dir    = plot_dir
        self.plot_format = plot_format
        self.plot_dpi    = plot_dpi
        self.plot_cmap   = plot_cmap
        
    def plot_gauss_blur(
            self
        ):
        logger.info(f"{self.name}-{self.date.date()} -> Plot Gaussian blur image.")
        fig, ax = plt.subplots(figsize=(self.width/500, self.height/500))
        im = ax.imshow(self.blurred, cmap='gray')
        ax.set(title=f"Gaussian blur ($\sigma$ = {self.sigma:.2f} px)", aspect='equal', xlabel='x [cm]', ylabel='y [cm]')
        fig.savefig(os.path.join(self.plot_dir, f"{self.field_id}_{self.date.date()}_03_gauss_blur"+self.plot_format), dpi=self.plot_dpi, bbox_inches='tight')
        fig.clf()
        plt.close("all")
        del fig, ax
    
    def plot_peaks(
            self
        ):
        logger.info(f"{self.name}-{self.date.date()} -> Plot peak position image.")
        fig, ax = plt.subplots(figsize=(self.width/500, self.height/500))
        ax.scatter(*self.peaks.T[::-1], color='red', s=2, label=f"{len(self.peaks)} peaks")
        ax.imshow(self.blurred, cmap=self.plot_cmap)
        ax.set(title=f"Peaks (min. distance = {self.min_peak_distance} cm = {self.min_peak_distance/self.px_res:.2f} px)", aspect='equal', xlabel='x [px]', ylabel='y [px]')
        ax.legend()
        fig.savefig(os.path.join(self.plot_dir, f"{self.field_id}_{self.date.date()}_04_peaks"+self.plot_format), dpi=self.plot_dpi, bbox_inches='tight')
        fig.clf()
        plt.close("all")
        del fig, ax
        
    def run(
            self,
            segmentation_mask: np.ndarray,
            #grow_function_cover_ratios: np.array,
            #dates: np.array,
            px_resolution: float,
            cover_ratio: float,
            date: dt.datetime,
            field_id: str,
            raster_meta: Dict
        ):
        self.date = date
        self.field_id = field_id
        self.px_res = px_resolution
        
        if len(segmentation_mask) > 0:
        
            # apply gaussian filter with scaled sigma
            if self.use_growfunction:
                raise NotImplementedError()
                #cover_ratio = grow_function_cover_ratios[dates == date]
                #logger.info(f"{self.name}-{self.date.date()} -> Use cover ratio from grow function fit. ({100.*cover_ratio:.2f} %)")
            else:
                logger.info(f"{self.name}-{self.date.date()} -> Use standard cover ratio. ({100.*cover_ratio:.2f} %)")

            self.sigma = (self.gauss_sigma_bounds[0] + cover_ratio*np.diff(self.gauss_sigma_bounds)[0]) / self.px_res

            logger.info(f"{self.name}-{self.date.date()} -> Blurring with sigma = {self.sigma*px_resolution:.2f} cm = {self.sigma:.2f} px.")
            self.blurred = gaussian(segmentation_mask.astype(np.float32), sigma=self.sigma)

            # detect peaks
            logger.info(f"{self.name}-{self.date.date()} -> Detect peaks with threshold {self.peak_threshold} and min. distance = {self.min_peak_distance} cm = {self.min_peak_distance/self.px_res:.2f} px.")
            self.peaks = peak_local_max(self.blurred, min_distance=int(np.round(self.min_peak_distance/self.px_res)), threshold_abs=self.peak_threshold, exclude_border=False)
            # convert peak position from pixel to cm coordinates with UTM coordinate transformation
            utm_peaks, utm_transform = px_to_utm(point_cloud=self.peaks, raster_meta=raster_meta) 
            utm_peaks *= 100 # m * 100 = cm
            n_peaks = len(self.peaks)
            self.height, self.width = self.blurred.shape
            logger.info(f"{self.name}-{self.date.date()} -> {n_peaks} peaks detected.")

            if (self.make_orthoimage):
                makeDirectory(self.orthoimage_dir)
                logger.info(f"{self.name}-{self.date.date()} -> Save Gauss blurred orthoimage.")
                write_onechannel_raster(os.path.join(self.orthoimage_dir, f"{self.field_id}_{self.date.date()}_blurred.tif"),
                                        self.blurred,
                                        raster_meta,
                                        "float32")

                logger.info(f"{self.name}-{self.date.date()} -> Export found peak positions as KML file.")
                kml = simplekml.Kml()
                for (lon, lat) in np.asarray(xy(raster_meta["transform"], *self.peaks.T)).T:
                    kml.newpoint(coords=[(lon, lat)])
                kml.save(os.path.join(self.orthoimage_dir, f"{self.field_id}_{self.date.date()}_peaks.kml"))

        else:
            logger.warn(f"{self.name}-{self.date.date()} -> No segmentation mask due to large cover ratio -> Skip plot.")
            utm_peaks = np.array([])
            # calculate UTM zone
            lon, lat = np.asarray(xy(raster_meta["transform"], raster_meta["height"]//2, raster_meta["width"]//2))
            utm_zone = int(np.floor((lon/360)*60+31))
            utm_transform = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84')

        self.save(obj=utm_peaks, name="plant_positions", type_="pickle")
        self.save(obj=utm_transform, name="utm_transform", type_="pickle")

        # plot blurred image and contrast image with peak positions
        if (len(segmentation_mask) > 0) and self.plot_result:
            makeDirectory(self.plot_dir)
            self.plot_gauss_blur()
            self.plot_peaks()
            gc.collect()
        

class LoadPeaks(Task):
    
    def __init__(
            self,
            field_id: str,
            plot_result: bool,
            plot_dir: str,
            plot_format: str,
            plot_dpi: int,
            plot_cmap: str
        ):
        super().__init__()
        self.field_id    = field_id
        self.plot_result = plot_result
        self.plot_dir    = plot_dir
        self.plot_format = plot_format
        self.plot_dpi    = plot_dpi
        self.plot_cmap   = plot_cmap
        
    def plot(
            self
        ):
        logger.info(f"{self.name} -> Plot raw peaks image.")
        fig, ax = plt.subplots()
        ax.scatter(*self.C.T, s=2, alpha=0.8, c=self.layers, cmap=self.plot_cmap)
        ax.set(title=f"{self.field_id}\nraw points", xlabel='x [cm]', ylabel='y [cm]', aspect='equal')
        fig.savefig(os.path.join(self.plot_dir, f"{self.field_id}_01_raw"+self.plot_format), dpi=self.plot_dpi, bbox_inches='tight')
        fig.clf()
        plt.close("all")
        del fig, ax
        
    def run(
            self,
            reduced_results: List[Dict[str, Dict]]
        ):
        cover_ratios, dates, gps_transforms, px_resolutions, field_ids, peaks, utm_transforms, segmentation_masks = [], [], [], [], [], [], [], []
        
        for r in reduced_results:
            try:
                if len(r["config"].keys()) == 1:
                    cover_ratios.append(r["result"]["cover_ratio"])
                    dates.append(r["result"]["date"])
                    gps_transforms.append(r["result"]["raster_meta"]["transform"])
                    px_resolutions.append(r["result"]["px_resolution"])
                    field_ids.append(r["result"]["field_id"])
                    segmentation_masks.append(r["result"]["segmentation_mask"])
                else:
                    peaks.append(r["result"]["plant_positions"])
                    utm_transforms.append(r["result"]["utm_transform"])
            except:
                logger.error(r)

            
        assert len(np.unique(field_ids)) == 1, logger.error(f"{self.name} -> Multiple field IDs!")
        assert np.unique(field_ids)[0] == self.field_id, logger.error(f"{self.name} -> Wrong field ID!")
        
        cover_ratios = np.asarray(cover_ratios)
        px_resolutions = np.asarray(px_resolutions)
        dates = pd.DatetimeIndex(dates)
        P = np.asarray(peaks)
        
        logger.info(f"{self.name} -> Load data for {len(dates)} dates.")
        
        # sort dates and layers by cover ratio
        cr_sort = np.argsort(cover_ratios)
        P = P[cr_sort]
        dates = dates[cr_sort]
        segmentation_masks = [segmentation_masks[c] for c in cr_sort]
        gps_transforms = [gps_transforms[c] for c in cr_sort]
        px_resolutions = px_resolutions[cr_sort]
        cover_ratios = np.sort(cover_ratios)
        n_layers = len(dates)

        logger.info(f"{self.name} -> Sorted dates and layers by cover ratio. Layers: {cr_sort}, dates: {dates}, cover ratios: {cover_ratios}")
        
        # dates for printing (e.g. in plots)
        printdates = dates.format(formatter=lambda x: x.strftime('%m-%d'))

        emptymask = [len(p)>0 for p in P]
        logger.info(f"{self.name} -> Peaks for {np.sum(emptymask)} dates available.")
        
        # stack point clouds and save layers
        self.C = np.vstack(P[emptymask])
        self.layers = np.repeat(np.arange(len(P)), np.array([len(p) for p in P]))

        self.save(obj=self.C, name="point_cloud", type_="pickle")
        self.save(obj=self.layers, name="layers", type_="pickle")
        self.save(obj=cover_ratios, name="cover_ratios", type_="pickle")
        self.save(obj=self.field_id, name="field_id", type_="json")
        self.save(obj=printdates, name="printdates", type_="pickle")
        self.save(obj=dates, name="dates", type_="pickle")
        self.save(obj=gps_transforms, name="gps_transforms", type_="pickle")
        self.save(obj=px_resolutions, name="px_resolutions", type_="pickle")
        self.save(obj=utm_transforms, name="utm_transforms", type_="pickle")
        self.save(obj=segmentation_masks, name="segmentation_masks", type_="pickle")
          
        # plot raw point information
        if self.plot_result:
            makeDirectory(self.plot_dir)
            self.plot()
            gc.collect()

    
class AlignPoints(Task):
    
    def __init__(
            self,
            max_centroid_distance_cpd: float,
            max_centroid_distance_group: float,
            make_orthoimage: bool,
            orthoimage_dir: str,
            plot_result: bool,
            plot_dir: str,
            plot_format: str,
            plot_dpi: int,
            plot_cmap: str
        ):
        super().__init__()
        self.max_centroid_distance_cpd = max_centroid_distance_cpd
        self.max_centroid_distance_group = max_centroid_distance_group
        self.make_orthoimage = make_orthoimage
        self.orthoimage_dir = orthoimage_dir
        self.plot_result = plot_result
        self.plot_dir    = plot_dir
        self.plot_format = plot_format
        self.plot_dpi    = plot_dpi
        self.plot_cmap   = plot_cmap
        
    @staticmethod
    def transform(
            coords: np.array,
            T: np.array
        ) -> np.array:
        return T[0]*coords@T[1] + T[2]
    
    def plot_aligned(
            self
        ):
        logger.info(f"{self.name} -> Plot aligned peak position image.")
        fig, ax = plt.subplots()
        ax.scatter(*self.P_aligned.T, s=2, alpha=0.8, c=self.layers, cmap=self.plot_cmap)
        ax.set(title=f"{self.field_id}\naligned points\naligned dates: {self.aligned_dates}", xlabel='x - mean [cm]', ylabel='y - mean [cm]', aspect='equal')
        fig.savefig(os.path.join(self.plot_dir, f"{self.field_id}_02_aligned"+self.plot_format), dpi=self.plot_dpi, bbox_inches='tight')
        fig.clf()
        plt.close("all")
        del fig, ax
        
    def plot_confidence(
            self
        ):
        logger.info(f"{self.name} -> Plot alignment mean confidence.")
        fig, ax = plt.subplots()
        ax.scatter(100*self.cover_ratios, 100*self.median_conf)
        ax.set(xlim=(0,100), ylim=(0,100), title=f"{self.field_id}\n", xlabel='cover ratio [%]', ylabel='median alignment confidence [%]', aspect='equal')
        ax.grid()
        fig.savefig(os.path.join(self.plot_dir, f"{self.field_id}_03_cr_vs_conf"+self.plot_format), dpi=self.plot_dpi, bbox_inches='tight')
        fig.clf()
        plt.close("all")
        del fig, ax
        
    def run(
            self,
            point_cloud: np.ndarray,
            layers: np.array,
            cover_ratios: np.array,
            printdates: np.array,
            field_id: str,
            utm_transforms: List
        ):
        self.field_id = field_id
        self.layers = layers
        self.printdates = printdates
        self.cover_ratios = cover_ratios

        uni_layers = np.sort(np.unique(layers))
        n_layers = len(self.cover_ratios)

        # centralize point clouds
        # calculate centroid of all points in UTM coordinates
        P_mean = point_cloud.mean(axis=0)
        # apply on point cloud
        P_c = point_cloud - P_mean

        scaF = np.ones(n_layers)
        rotA = np.zeros(n_layers)
        traV = np.zeros((n_layers, 2))
        self.median_conf = np.nan*np.ones(n_layers)

        self.P_aligned = P_c.copy()
        P_centroid = P_c[layers == uni_layers[0]]
        self.P_aligned[layers == uni_layers[0]] = P_centroid

        aligned_layers = []
        
        for l in uni_layers:
            if l != 0:
                X = P_centroid
                Y = P_c[layers == l]

                # filter points with no neighbours inside max_dist radius
                nnX = NearestNeighbors(n_neighbors=1, n_jobs=-1)
                nnY = NearestNeighbors(n_neighbors=1, n_jobs=-1)
                nnX.fit(X)
                nnY.fit(Y)
                distXY, _ = nnY.kneighbors(X)
                distYX, _ = nnX.kneighbors(Y)
                X_filt = X[(distXY <= self.max_centroid_distance_cpd).flatten()]
                Y_filt = Y[(distYX <= self.max_centroid_distance_cpd).flatten()]

                # Rigid Transformation: T(X) = s*R@X + t
                # s: scaling factor
                # R: rotation matrix
                # t: translation vector
                # A. Myronenko, X. Song: "Point Set Registration: Coherent Point Drift"
                # https://arxiv.org/pdf/0905.2635.pdf

                # registration with filtered points
                logger.info(f"{self.name} -> Layer {l} of {len(uni_layers)} -> Try to align {len(Y_filt)} of {len(Y)} points to {len(X_filt)} of {len(X)} centroids. Maximum centroid distance: {self.max_centroid_distance_cpd} cm.")
                reg = RigidRegistration(X=X_filt, Y=Y_filt) # X = target, Y = source
                _, T = reg.register()
                
                self.median_conf[l] = np.median(np.max(reg.P, axis=1))
                # if registration was confident (median confidence above 68%) accept, else discard

                #if self.median_conf[l] > 0.68:
                scaF[l] = T[0]
                rotA[l] = np.rad2deg(np.arccos(T[1][0,0]))
                traV[l] = T[2]
                self.P_aligned[layers == l] = self.transform(Y, T)
                aligned_layers.append(l)
                logger.info(f"{self.name} -> Layer {l} of {len(uni_layers)} alignable layers aligned. Scaling factor: {scaF[l]}. Rotation angle: {rotA[l]} °. Translation vector: {traV[l]} cm. Median confidence: {100.*self.median_conf[l]:.2f} %")

                #else:
                #    logger.warn(f"{self.name} -> Layer {l} of {len(uni_layers)} has too low median confidence ({100.*self.median_conf[l]:.2f} %). Layer will not be aligned.")

                #if l <= self.max_reference_layer:
                logger.info(f"{self.name} -> Layer {l} of {len(uni_layers)} -> Group with maximum centroid distance: {self.max_centroid_distance_group} cm.")
                _, P_centroid = group_points(self.P_aligned[self.layers <= l],
                                             self.layers[self.layers <= l],
                                             max_dist=self.max_centroid_distance_group)              

        logger.info(f"{self.name} -> All points aligned.")      
                
        self.save(obj=self.P_aligned, name="point_cloud_aligned", type_="pickle")
        self.save(obj=P_mean, name="point_cloud_mean", type_="pickle")
        self.save(obj=(scaF, rotA, traV, self.median_conf), name="align_transform", type_="pickle")
        
        self.aligned_dates = np.asarray(self.printdates)[aligned_layers].tolist()
        
        if self.plot_result:
            makeDirectory(self.plot_dir)
            self.plot_aligned()
            self.plot_confidence()
            gc.collect()

        if (self.make_orthoimage):
            makeDirectory(self.orthoimage_dir)
            logger.info(f"{self.name} -> Export aligned point cloud as KML file.")
            kml = simplekml.Kml()
            for l in uni_layers:
                folder = kml.newfolder(name=self.printdates[l])
                for (lon, lat) in np.asarray(utm_transforms[l](*((self.P_aligned[self.layers == l]+P_mean)/100.).T, inverse=True)).T:
                    folder.newpoint(coords=[(lon, lat)])
            kml.save(os.path.join(self.orthoimage_dir, f"{self.field_id}_peaks_aligned.kml"))
        

class AlignCroplines(Task):
    
    def __init__(
            self,
            plot_result: bool,
            plot_dir: str,
            plot_format: str,
            plot_dpi: int,
            plot_cmap: str
        ):
        super().__init__()
        self.plot_result = plot_result
        self.plot_dir    = plot_dir
        self.plot_format = plot_format
        self.plot_dpi    = plot_dpi
        self.plot_cmap   = plot_cmap
        
    @staticmethod
    def rotation2d(
            deg: float
        ) -> np.array:
        a = np.deg2rad(deg)
        return np.array([[np.cos(a),  -np.sin(a)],
                         [np.sin(a), np.cos(a)]])

    def findHoughAnglesNested(
            self,
            image: np.ndarray,
            i_max: int,
            steps: int,
            bin_tolerance: int
        ) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        test_angles = np.linspace(-np.pi/2, np.pi/2, steps, endpoint=False)
        mean, std = 0, np.pi/2
        for i in range(i_max):
            logger.info(f"{self.name} -> Iteration {i}/{i_max} -> Perform Hough transform for {steps} angles in [{np.rad2deg(test_angles.min())}, {np.rad2deg(test_angles.max())}]°.")
            h, theta, d = hough_line(image, theta=test_angles)
            _, angles, dists = hough_line_peaks(h, theta, d)

            hist, bins = np.histogram(angles, bins=steps, range=(test_angles.min(), test_angles.max()))
            mean = np.mean(angles)
            std = np.std(angles, ddof=1)

            a_min = bins[np.max((0, np.argmax(hist)-bin_tolerance))]
            a_max = bins[np.min((steps, np.argmax(hist)+1+bin_tolerance))]
            test_angles = np.linspace(a_min, a_max, steps)
            
            if np.all(np.mean(angles) == angles):
                logger.info(f"{self.name} -> Iteration {i}/{i_max} -> Terminate! Best alpha = {np.rad2deg(mean):.4f} °.")
                return (angles, dists, h, theta, d)
            else:
                logger.info(f"{self.name} -> Iteration {i}/{i_max} -> alpha = ({np.rad2deg(mean):.4f} +/- {np.rad2deg(std):.4f}) °.")
        logger.info(f"{self.name} -> Best alpha after {i_max} iterations = ({np.rad2deg(mean):.4f} +/- {np.rad2deg(std):.4f}) °.")
        return (angles, dists, h, theta, d)
    
    def plot(
            self
        ):
        logger.info(f"{self.name} -> Plot cropline rotation.")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ax = axes.ravel()

        ax[0].imshow(self.hough_img, cmap=self.plot_cmap)
        ax[0].set_title('image')

        ax[1].imshow(self.hough_img, cmap=self.plot_cmap)
        ax[1].set_ylim((self.hough_img.shape[0], 0))
        ax[1].set_title('detected lines')

        for angle, dist in zip(self.angles, self.dists):
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            ax[1].axline((x0, y0), slope=np.tan(angle + np.pi/2))

        fig.savefig(os.path.join(self.plot_dir, f"{self.field_id}_04_rot_angle"+self.plot_format), dpi=self.plot_dpi, bbox_inches='tight')
        fig.clf()
        plt.close("all")
        del fig, ax
        
    def run(
            self,
            point_cloud_aligned: np.ndarray,
            printdates: np.array,
            field_id: str
            ):
        self.field_id = field_id
        point_cloud = point_cloud_aligned
        self.printdates = printdates
        
        # Hough transform with fixed resolution (cm/px)
        res = 1 # cm/px
        logger.info(f"{self.name} -> Bin point cloud into image with resolution {res} cm/px.")
        self.hough_img, _, _ = np.histogram2d(*point_cloud.T,
                                bins=[
                                    np.arange(point_cloud[:,0].min(), point_cloud[:,0].max(), res),
                                    np.arange(point_cloud[:,1].min(), point_cloud[:,1].max(), res)
                                    ])

        # perform iterative Hough line detection with nested intervals method
        i_max = 50
        steps = 180
        bin_tolerance = 2
        self.angles, self.dists, self.h, self.theta, self.d = self.findHoughAnglesNested(self.hough_img, i_max, steps, bin_tolerance)
        self.alpha_best = np.rad2deg(np.mean(self.angles))
        self.alpha_best_std = np.rad2deg(np.std(self.angles, ddof=1))
        
        # median cropline distance
        d_cl_median = np.median(np.diff(np.sort(self.dists))) * res # px * (cm/px) = cm

        coords_rot = (self.rotation2d(self.alpha_best)@point_cloud.T).T
        
        logger.info(f"{self.name} -> Croplines rotated with best angle: ({self.alpha_best:.4f} +/- {self.alpha_best_std:.4f}) °. Median cropline distance: {d_cl_median:.4f} cm.")
        
        self.save(obj=coords_rot, name="point_cloud_rotated", type_="pickle")
        self.save(obj=self.alpha_best, name="rotation_angle", type_="json")
        self.save(obj=d_cl_median, name="median_cropline_distance", type_="json")

        if self.plot_result:
            makeDirectory(self.plot_dir)
            self.plot()
            gc.collect()   

class FindCroplines(Task):
    
    def __init__(
            self,
            plot_result: bool,
            plot_dir: str,
            plot_format: str,
            plot_dpi: int
        ):
        super().__init__()
        self.plot_result = plot_result
        self.plot_dir    = plot_dir
        self.plot_format = plot_format
        self.plot_dpi    = plot_dpi
    
    def plot_peaks(
            self
        ):
        logger.info(f"{self.name} -> Plot cropline peak positions.")
        fig, ax = plt.subplots()
        ax.plot(self.y_test, self.incl_points_sum)
        ax.scatter(self.y_test[self.peak_pos], self.incl_points_sum[self.peak_pos], s=20, c='r', label=f"{len(self.peak_pos)} peaks")
        ax.set(xlabel='position of window center (y-coords of rotated points)', ylabel='points inside window',
               xlim=(self.Y.min()-self.scan_window, self.Y.max()+self.scan_window), ylim=(0,None))
        ax.legend()
        ax.set(title=f"{self.field_id}\ncropline peaks")
        fig.savefig(os.path.join(self.plot_dir, f"{self.field_id}_05_cropline_peaks"+self.plot_format), dpi=self.plot_dpi, bbox_inches='tight')
        fig.clf()
        plt.close("all")
        del fig, ax
        
    def plot_croplines(self): 
        logger.info(f"{self.name} -> Plot rotated points with marked croplines.")
        fig, ax = plt.subplots()
        ax.scatter(*self.point_cloud.T, s=2, alpha=1, c="C0")
        ax.hlines(self.croplines_ypos, xmin = self.point_cloud[:,0].min(), xmax = self.point_cloud[:,0].max(), color='r')
        ax.set(title=f"{self.field_id}\nrotated points with croplines", xlabel='x - mean (rotated)', ylabel='y - mean (rotated)', aspect='equal')
        fig.savefig(os.path.join(self.plot_dir, f"{self.field_id}_06_croplines"+self.plot_format), dpi=self.plot_dpi, bbox_inches='tight')
        fig.clf()
        plt.close("all")
        del fig, ax
        
    def run(
            self,
            point_cloud_rotated: np.ndarray,
            field_id: str,
            median_cropline_distance: float,
            px_resolutions: np.ndarray
        ):
        self.field_id    = field_id
        self.point_cloud = point_cloud_rotated
        self.Y = self.point_cloud[:,1]

        scan_resolution = 10000 # steps per cropline

        self.scan_window = median_cropline_distance / 10
        self.scan_precision = median_cropline_distance / scan_resolution

        logger.info(f"{self.name} -> Given cropline distance estimate of {median_cropline_distance} cm results in a scan window of {self.scan_window} cm and precision of {self.scan_precision} cm.")

        self.y_test = np.arange(self.Y.min()-self.scan_window, self.Y.max()+self.scan_window, self.scan_precision)
        incl_points_sum = []

        for y_center in self.y_test:
            incl_points_sum.append(np.sum((self.Y >= y_center-(self.scan_window/2)) & (self.Y <= y_center+(self.scan_window/2))))
        self.incl_points_sum = np.asarray(incl_points_sum)

        self.peak_pos = find_peaks(self.incl_points_sum, distance=int(0.75*scan_resolution))[0]

        self.croplines_ypos = self.y_test[self.peak_pos]

        logger.info(f"{self.name} -> {len(self.croplines_ypos)} croplines found: {self.croplines_ypos}")
        
        self.save(obj=self.croplines_ypos, name="croplines_ypos", type_="pickle")

        if self.plot_result:
            makeDirectory(self.plot_dir)
            self.plot_peaks()
            self.plot_croplines()
            gc.collect()


class FilterWeed(Task):
    
    def __init__(
            self,
            threshold_factor: float,
            plot_result: bool,
            plot_dir: str,
            plot_format: str,
            plot_dpi: int
        ):
        super().__init__()
        self.threshold_factor = threshold_factor
        self.plot_result = plot_result
        self.plot_dir    = plot_dir
        self.plot_format = plot_format
        self.plot_dpi    = plot_dpi
        
    @staticmethod
    def find_nearest(
            array: np.array,
            values: np.array
        ) -> np.array:
        indices = np.abs(np.subtract.outer(array, values)).argmin(axis=0)
        return array[indices]    
        
    def plot(
            self
        ):
        logger.info(f"{self.name} -> Plot point cloud with masked weed.")
        fig, ax = plt.subplots()
        ax.scatter(*self.point_cloud_aligned_filtered.T, s=5, alpha=1, label="valid")
        ax.scatter(*self.point_cloud_aligned[~self.weedmask].T, s=5, alpha=1, color='r', label=f"Weed ({self.weed_percentage:.2f} %)")
        ax.set(title=f"{self.field_id}\nmasked weed", xlabel='x - mean [cm]', ylabel='y - mean [cm]', aspect='equal')
        ax.legend()
        fig.savefig(os.path.join(self.plot_dir, f"{self.field_id}_07_weed_mask"+self.plot_format), dpi=self.plot_dpi, bbox_inches='tight')
        fig.clf()
        plt.close("all")
        del fig, ax
        
    def run(
            self,
            point_cloud_rotated: np.ndarray,
            point_cloud_aligned: np.ndarray,
            point_cloud: np.ndarray,
            layers: np.array,
            croplines_ypos: np.array,
            field_id: str
        ):
        self.field_id = field_id
        self.point_cloud_aligned = point_cloud_aligned
        median_line_distance = np.median(np.diff(croplines_ypos))
        next_line_distance = np.abs(point_cloud_rotated[:,1] - self.find_nearest(croplines_ypos, point_cloud_rotated[:,1]))   

        logger.info(f"{self.name} -> Calculated median seeding line distance: {median_line_distance:.2f} cm. Masking weed with threshold factor {self.threshold_factor}.")
        self.weedmask = next_line_distance <= self.threshold_factor*median_line_distance
        self.weed_percentage = 100*np.sum(~self.weedmask)/len(point_cloud_aligned)

        if self.weed_percentage < 30:
            logger.info(f"{self.name} -> {np.sum(~self.weedmask)} points masked as weed ({self.weed_percentage:.2f} %).")
        else:
            logger.warn(f"{self.name} -> High percentage of points masked as weed ({self.weed_percentage:.2f} %). There might be an error in the analysis.")

        self.point_cloud_aligned_filtered, point_cloud_rotated_filtered, point_cloud_filtered, layers_filtered = point_cloud_aligned[self.weedmask], point_cloud_rotated[self.weedmask], point_cloud[self.weedmask], layers[self.weedmask]     
     
        self.save(obj=self.weedmask, name="weedmask", type_="pickle")
        self.save(obj=self.point_cloud_aligned_filtered, name="point_cloud_aligned_weedfiltered", type_="pickle")
        self.save(obj=point_cloud_rotated_filtered, name="point_cloud_rotated_weedfiltered", type_="pickle")
        self.save(obj=point_cloud_filtered, name="point_cloud_weedfiltered", type_="pickle")
        self.save(obj=layers_filtered, name="layers_weedfiltered", type_="pickle")           
        
        if self.plot_result:
            makeDirectory(self.plot_dir)
            self.plot()
            gc.collect()


class GroupPoints(Task):
    
    def __init__(
            self,
            max_centroid_distance: float
        ):
        super().__init__()
        self.max_centroid_distance = max_centroid_distance
        
    def run(
            self,
            point_cloud_weedfiltered: np.array,
            point_cloud_aligned_weedfiltered: np.array,
            point_cloud_rotated_weedfiltered: np.array,
            layers_weedfiltered: np.array
        ):  
        labels, centroids = group_points(point_cloud_aligned_weedfiltered, layers_weedfiltered, max_dist=self.max_centroid_distance)
        labels_dist = np.bincount(np.bincount(labels[labels>=0]))[1:]

        logger.info(f"{self.name} -> {labels.max()+1} groups found with distribution {labels_dist}, {np.sum(labels==-1)}/{len(labels)} points discarded.")

        # filter discarded points out
        point_cloud_aligned_weedfiltered = point_cloud_aligned_weedfiltered[labels>=0]
        point_cloud_rotated_weedfiltered = point_cloud_rotated_weedfiltered[labels>=0]
        point_cloud_weedfiltered = point_cloud_weedfiltered[labels>=0]
        layers_weedfiltered = layers_weedfiltered[labels>=0]
        labels = labels[labels>=0]
        
        self.save(obj=point_cloud_weedfiltered, name="point_cloud_weedfiltered_grouped", type_="pickle")
        self.save(obj=point_cloud_aligned_weedfiltered, name="point_cloud_aligned_weedfiltered_grouped", type_="pickle")
        self.save(obj=point_cloud_rotated_weedfiltered, name="point_cloud_rotated_weedfiltered_grouped", type_="pickle")
        self.save(obj=labels, name="group_labels", type_="pickle")
        self.save(obj=layers_weedfiltered, name="layers_weedfiltered_grouped", type_="pickle")


class SortGroupLabels(Task):
    
    def __init__(
            self,
            plot_result: bool,
            plot_dir: str,
            plot_format: str,
            plot_dpi: int,
            plot_cmap: str
        ):
        super().__init__()
        self.plot_result = plot_result
        self.plot_dir    = plot_dir
        self.plot_format = plot_format
        self.plot_dpi    = plot_dpi
        self.plot_cmap   = plot_cmap
        
    @staticmethod
    def centroid(
            points
        ):
        return points.mean(axis=0)
        
    @staticmethod
    def find_nearest_index(
            array,
            values
        ):
        indices = np.abs(np.subtract.outer(array, values)).argmin(axis=0)
        return indices    
        
    def plot(
            self
        ):
        logger.info(f"{self.name} -> Plot sorted and unsorted group labels.")
        fig, ax = plt.subplots(1, 2, sharey=True)
        ax[0].scatter(self.point_cloud[:,0], self.point_cloud[:,1], s=1, c=self.group_labels, alpha=0.6, cmap=self.plot_cmap)
        sc = ax[1].scatter(self.point_cloud[:,0], self.point_cloud[:,1], s=1, c=self.labels_sorted, alpha=0.6, cmap=self.plot_cmap)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("group ID")
        for a in ax:
            a.set(xlabel='x - mean [cm]', aspect='equal')
        ax[0].set(ylabel='y - mean [cm]')
        fig.suptitle(f"{self.field_id}\nsort group IDs")
        fig.savefig(os.path.join(self.plot_dir, f"{self.field_id}_08_sorted_labels"+self.plot_format), dpi=self.plot_dpi, bbox_inches='tight')
        fig.clf()
        plt.close("all")
        del fig, ax
        
    def run(
            self,
            field_id: str,
            point_cloud_rotated_weedfiltered_grouped: np.ndarray,
            group_labels: np.array,
            croplines_ypos: np.array
        ):
        
        self.point_cloud = point_cloud_rotated_weedfiltered_grouped
        self.group_labels = group_labels
        self.field_id = field_id
        
        self.labels_sorted = -1*np.ones_like(group_labels)

        group_centroids = np.array([self.centroid(self.point_cloud[group_labels == l]) for l in range(group_labels.max()+1)])
        group_cropline_ids = self.find_nearest_index(croplines_ypos, group_centroids[:,1])

        group_order = np.lexsort((group_centroids[:,0], group_cropline_ids))

        for l_old, l_new in enumerate(group_order):
            self.labels_sorted[group_labels == l_new] = l_old

        group_cropline_ids_sorted = group_cropline_ids[group_labels]
        _, group_sizes = np.unique(self.labels_sorted, return_counts=True)
        
        self.save(obj=self.labels_sorted, name="group_labels_sorted", type_="pickle")
        self.save(obj=group_cropline_ids_sorted, name="group_cropline_ids_sorted", type_="pickle")
        self.save(obj=group_sizes, name="group_sizes_sorted", type_="pickle")

        if self.plot_result:
            makeDirectory(self.plot_dir)
            self.plot()
            gc.collect()

 
class SavePlantsDataFrame(Task):
    
    def __init__(
            self,
            save_dir: str
        ):
        super().__init__()
        self.save_dir = save_dir     
        
    def run(
            self,
            field_id: str,
            dates: pd.DatetimeIndex,
            cover_ratios: np.array,
            gps_transforms: List,
            px_resolutions: np.array,
            utm_transforms: List,
            point_cloud_mean: np.ndarray,
            align_transform: Tuple[Union[np.array,np.ndarray]],
            rotation_angle: float,
            layers_weedfiltered_grouped: np.array,
            group_sizes_sorted: np.array,
            group_cropline_ids_sorted: np.array,
            point_cloud_weedfiltered_grouped: np.ndarray,
            point_cloud_aligned_weedfiltered_grouped: np.ndarray,
            group_labels_sorted: np.array,
            croplines_ypos: np.array
        ):
        # back-transform peak position data from cm (UTM) into GPS coordinates
        point_cloud_weedfiltered_grouped_gps = np.hstack([utm_transforms[l](*point_cloud_weedfiltered_grouped[layers_weedfiltered_grouped == l].T/100., inverse=True) for l in np.unique(layers_weedfiltered_grouped)]).T

        (scaF, rotA, traV, median_conf) = align_transform
        align_transform_ = np.vstack((scaF, rotA, traV[:,0], traV[:,1])).T # cm
        group_centroids = np.array([point_cloud_aligned_weedfiltered_grouped[group_labels_sorted == l].mean(axis=0) for l in range(group_labels_sorted.max()+1)]) # cm
        n_layers = len(dates)

        df_meta = pd.DataFrame()
        for i in range(len(dates)):
            df_meta = df_meta.append(
                dict([("field_id"                , field_id),
                      ("date"                    , dates.values[i]),
                      ("cover_ratio"             , cover_ratios[i]), # %
                      ("xy_center"               , point_cloud_mean), # cm (UTM)
                      ("align_median_confidence" , median_conf[i]), # %
                      ("align_transform"         , align_transform_[i]), # cm (UTM)
                      ("gps_transform"           , gps_transforms[i]), # px <-> lonlat
                      ("px_resolution"           , px_resolutions[i]), # cm/px
                      ("utm_transform"           , utm_transforms[i]), # m (UTM) <-> lonlat
                      ("rotation_angle"          , rotation_angle)]), ignore_index=True) # degree

        df_plants = pd.DataFrame()
        for i in range(len(group_labels_sorted)):
            
            df_plants = df_plants.append(
                dict([("field_id"                        , field_id),
                      ("date"                            , dates.values[layers_weedfiltered_grouped[i]]),
                      ("group_id"                        , group_labels_sorted[i]),
                      ("group_size"                      , group_sizes_sorted[group_labels_sorted[i]]),
                      ("group_cropline_id"               , group_cropline_ids_sorted[i]),
                      ("xy_cm"                           , point_cloud_weedfiltered_grouped[i]), # cm (UTM)
                      ("xy_px"                           , list(rowcol(gps_transforms[np.argmax(dates.values==dates.values[layers_weedfiltered_grouped[i]])], *point_cloud_weedfiltered_grouped_gps[i]))), # px 
                      ("lonlat"                          , point_cloud_weedfiltered_grouped_gps[i]), # lonlat
                      ("xy_centered_aligned_cm"          , point_cloud_aligned_weedfiltered_grouped[i]), # cm (UTM)
                      ("xy_centroid_centered_aligned_cm" , group_centroids[group_labels_sorted[i]]), # cm (UTM)
                      ("y_cropline_rotated_cm"           , croplines_ypos[group_cropline_ids_sorted[i]]), # cm (UTM)
                      ("centroid_dist_cm"                , np.sqrt(np.sum((point_cloud_aligned_weedfiltered_grouped[i]-group_centroids[group_labels_sorted[i]])**2))), # cm (UTM)
                      ("detected"                        , True)]), ignore_index=True)

        logger.info(f"{self.name} -> Detected plants added to DataFrame.")
        
        df_plants = df_plants.append(add_non_detected(df_plants[df_plants["group_size"] < n_layers], df_meta))

        df_plants["field_id"]          = df_plants["field_id"].astype(str)
        df_plants["group_id"]          = df_plants["group_id"].astype(int)
        df_plants["group_size"]        = df_plants["group_size"].astype(int)
        df_plants["group_cropline_id"] = df_plants["group_cropline_id"].astype(int)
        df_plants["detected"]          = df_plants["detected"].astype(bool)

        df_plants = df_plants.sort_values(by=["group_id", "date"], ignore_index=True)

        ndates = len(df_plants["date"].value_counts())
        logger.info(f"{self.name} -> Complemented DataFrame with non-detected plant positions. {ndates}/{len(dates.values)} dates available.")

        makeDirectory(self.save_dir)
        
        plants_save_path = os.path.join(self.save_dir, f"{field_id}_plants.pkl")
        meta_save_path   = os.path.join(self.save_dir, f"{field_id}_meta.pkl")

        try:
            df_plants.to_pickle(plants_save_path)
            logger.info(f"{self.name} -> DataFrame with plants saved at {plants_save_path}.")
            df_meta.to_pickle(meta_save_path)
            logger.info(f"{self.name} -> DataFrame with metadata saved at {meta_save_path}.")

        except:
            logger.error(f"{self.name} -> Could not save DataFrames.")
            
            
        self.save(obj="", name="_dummy", type_="json")
        

class EvaluateDetectionQuality(Task):

    def __init__(
            self,
            df_dir: str,
            image_dir: str,
            ground_truth_dir: str,
            image_channels: List[str],
            max_distance: float,
            save_dir: str,
            plot_result: bool,
            plot_dir: str,
            plot_format: str,
            plot_dpi: float
        ):
        super().__init__()
        self.df_dir = df_dir
        self.image_dir = image_dir
        self.ground_truth_dir = ground_truth_dir
        self.image_channels = np.asarray(image_channels)
        self.max_distance = max_distance
        self.save_dir = save_dir
        self.plot_result = plot_result
        self.plot_dir = plot_dir
        self.plot_format = plot_format
        self.plot_dpi = plot_dpi
   
    def plot(
            self
        ):
        logger.info(f"{self.name}-{self.date} -> Plot detections on image.")
        fig, ax = plt.subplots(figsize=(self.width/1000, self.height/1000))
        ax.imshow(self.img)
        if self.kml_filepath != "":
            ax.scatter(*self.gtxy.T[::-1], label=f"ground truth ({len(self.gtxy)})", s = 10, color="C0", alpha=0.5, marker="o")
        if self.pxy_direct != []:
            ax.scatter(*self.pxy_direct.T[::-1], label=f"direct detection ({len(self.pxy_direct)})", color="C2", s=1)
        if self.pxy_indirect != []:
            ax.scatter(*self.pxy_indirect.T[::-1], label=f"indirect detection ({len(self.pxy_indirect)})", color="C3", s=1)
        ax.legend()
        if self.kml_filepath != "":
            ax.set(title = f"{self.field_id}@{self.date}\nRecall = {100 * self.TP/(self.TP+self.FN):.2f} %\nPrecision = {100 * self.TP/(self.TP+self.FP):.2f} %")
            fig.savefig(os.path.join(self.plot_dir, f"{self.field_id}_{self.date}_detections_gt"+self.plot_format), dpi=self.plot_dpi, bbox_inches='tight')
        else:
            ax.set(title = f"{self.field_id}@{self.date}")
            fig.savefig(os.path.join(self.plot_dir, f"{self.field_id}_{self.date}_detections"+self.plot_format), dpi=self.plot_dpi, bbox_inches='tight')
        fig.clf()
        plt.close("all")
        del fig, ax

    def run(
            self,
            field_id: str,
            dates: pd.DatetimeIndex,
            gps_transforms: List,
            utm_transforms: List,
            px_resolutions: np.array
        ):
        self.field_id = field_id
        px_res = np.mean(px_resolutions)

        plants_all = pd.read_pickle(os.path.join(self.df_dir, f"{self.field_id}_plants.pkl"))
        #meta      = pd.read_pickle(os.path.join(self.df_dir, f"{self.field_id}_meta.pkl"))

        # filter out all inderect detections before the first detection happend
        drop_ind = []
        sorted_dates = sorted(dates)

        for g_id in np.unique(plants_all.group_id):
            group = plants_all[plants_all.group_id == g_id]
            first_detection_date = sorted_dates[group.detected.argmax()]
            drop_ind.extend(group.index[group.date < first_detection_date])

        plants = plants_all.drop(drop_ind)
        logger.info(f"{self.name} -> Filtered out leading indirect detections: {len(plants)}/{len(plants_all)} ({100.*len(plants)/len(plants_all):.2f} %) remaining.")
        del plants_all

        results = dict()

        # iterate over all dates
        for date, utm_transform, gps_transform in zip(dates, utm_transforms, gps_transforms):

            self.date = date._date_repr
            filedate = self.date.replace("-","")

            logger.info(f"{self.name}-{self.date} -> Calculate detection quality.")

            # retrieve image and shape file, if available
            kml_filepath = list(glob(f"{self.ground_truth_dir}/{field_id}_{filedate}*.kml"))
            tif_filepath = list(glob(f"{self.image_dir}/{field_id}_{filedate}*.tif"))[0]
            
            if len(kml_filepath) > 1:
                logger.warn(f"{self.name}-{self.date} -> Multiple ground truth shape files found for image {os.path.basename(tif_filepath)}. ",
                            f"Take first one in list: {os.path.basename(kml_filepath[0])}.")
                self.kml_filepath = kml_filepath[0]
            elif len(kml_filepath) == 1:
                logger.info(f"{self.name}-{self.date} -> Ground truth shape file found.")
                self.kml_filepath = kml_filepath[0]
            else:
                logger.warn(f"{self.name}-{self.date} -> No ground truth shape file found.")
                self.kml_filepath = ""
            
            # if ground truth data available, load positions
            if self.kml_filepath != "":
                try:
                    gtlatlon = readCoordsFromKml(self.kml_filepath)
                    gtutm = np.asarray(utm_transform(*gtlatlon.T)).T
                    self.gtxy = np.asarray(rowcol(gps_transform, xs=gtlatlon[:,0], ys=gtlatlon[:,1], op=lambda x: x)).T
                except Exception as e:
                    logger.warn(f"{self.name}-{self.date} -> Could not load shape file. Error: {e}. Continue without ground truth data.")
                    gtutm = []
                    self.gtxy = []
                    self.kml_filepath = ""
            else:
                gtutm = []
                self.gtxy = []
            
            # load indirect and (if available direct) detections
            try:
                self.pxy_indirect = np.vstack(plants[(plants["field_id"] == field_id) & (plants["date"] == date) & (plants["detected"]==False)]["xy_px"].values)
                plonlat_indirect = np.vstack(plants[(plants["field_id"] == field_id) & (plants["date"] == date) & (plants["detected"]==False)]["lonlat"].values)
            except:
                self.pxy_indirect = []
                plonlat_indirect = []
            try:
                self.pxy_direct = np.vstack(plants[(plants["field_id"] == field_id) & (plants["date"] == date) & (plants["detected"]==True)]["xy_px"].values)
                plonlat_direct = np.vstack(plants[(plants["field_id"] == field_id) & (plants["date"] == date) & (plants["detected"]==True)]["lonlat"].values)
            except:
                self.pxy_direct = []
                plonlat_direct = []
            
            if (plonlat_indirect != []) and (plonlat_direct != []):
                plonlat = np.vstack((plonlat_indirect, plonlat_direct))
            elif plonlat_indirect != []:
                plonlat = plonlat_indirect
            else:
                plonlat = plonlat_direct

            pxy_utm = np.asarray(utm_transform(*plonlat.T)).T

            # initalize results dictionary
            results[self.date] = {
                "true_positive":  np.nan,
                "false_positive": np.nan,
                "false_negative": np.nan
            }   
            
            # connect detection with ground truth and extract true/false positives and false negatives
            if self.kml_filepath != "":

                logger.info(f"{self.name}-{self.date} -> Compare detections with ground truth plant positions (max. tolerance radius: {self.max_distance} cm.")
                
                nn = NearestNeighbors(n_neighbors=1).fit(gtutm)
                dist, ind = map(lambda x: x.flatten(), nn.kneighbors(pxy_utm))

                self.TP, self.FP, self.FN = 0, 0, 0

                for i in range(len(gtutm)):
                    i_dist = dist[ind == i]
                    in_radius = i_dist <= self.max_distance/100.
                    if np.sum(in_radius) > 0:
                        self.TP += 1
                        self.FP += len(i_dist) - 1
                    else:
                        self.FN += 1
                        self.FP += len(i_dist)

                results[self.date]["true_positive"] = self.TP
                results[self.date]["false_positive"] = self.FP
                results[self.date]["false_negative"] = self.FN

            if self.plot_result:

                self.img = read_raster(tif_filepath, self.image_channels, ["R", "G", "B"])
                self.img /= np.nanmax(self.img)

                self.height, self.width, n_channels = self.img.shape

                makeDirectory(self.plot_dir)
                self.plot()
                del self.img
                gc.collect()

        # write results to a DataFrame
        logger.info(f"{self.name} -> Write results to DataFrame.")

        quality_df = pd.DataFrame()

        for date, values in results.items():
            quality_df = quality_df.append(
                dict([("field_id"       , self.field_id),
                      ("date"           , date),
                      ("true_positive"  , values["true_positive"]),
                      ("false_positive" , values["false_positive"]),
                      ("false_negative" , values["false_negative"])]), ignore_index=True)
            
        quality_df["precision"] = 100 * quality_df["true_positive"]/(quality_df["true_positive"]+quality_df["false_positive"])
        quality_df["recall"] = 100 * quality_df["true_positive"]/(quality_df["true_positive"]+quality_df["false_negative"])

        quality_df["true_positive"] = quality_df["true_positive"].apply(lambda x: int(x) if pd.notna(x) else x)
        quality_df["false_positive"] = quality_df["false_positive"].apply(lambda x: int(x) if pd.notna(x) else x)
        quality_df["false_negative"] = quality_df["false_negative"].apply(lambda x: int(x) if pd.notna(x) else x)
    
        quality_df["date"] = pd.DatetimeIndex(quality_df["date"])

        quality_df = quality_df.sort_values(["date"]) 

        quality_save_path = os.path.join(self.save_dir, f"{self.field_id}_det_quality.pkl")
        try:
            quality_df.to_pickle(quality_save_path)
            logger.info(f"{self.name} -> DataFrame with detection results saved at {quality_save_path}.")
        except:
            logger.error(f"{self.name} -> Could not save DataFrame.")

        self.save(obj=results, name="detection_quality", type_="json")
        self.save(obj=self.pxy_direct, name="direct_detections_xy", type_="pickle")
        self.save(obj=self.pxy_indirect, name="indirect_detections_xy", type_="pickle")
        self.save(obj=self.gtxy, name="ground_truth_xy", type_="pickle")


class MergePlantsDataFrame(Task):
    
    def __init__(
            self,
            save_dir: str
        ):
        super().__init__()
        self.save_dir = save_dir
        
    def run(
            self,
            reduced_results: List[Dict[str, Dict]]
        ):        
        plants_df_paths  = sorted(glob(os.path.join(self.save_dir, f"*_plants.pkl")))
        meta_df_paths    = sorted(glob(os.path.join(self.save_dir, f"*_meta.pkl")))
        quality_df_paths = sorted(glob(os.path.join(self.save_dir, f"*_det_quality.pkl")))
        
        logger.info(f"{self.name} -> Merge DataFrames of plants, metadata, and detection quality.")

        plants_df = pd.DataFrame()
        for p in plants_df_paths:
            plants_df = plants_df.append(pd.read_pickle(p), ignore_index=True)

        df_save_path = os.path.join(self.save_dir, "plants.pkl")
        plants_df.to_pickle(df_save_path)
        logger.info(f"{self.name} -> Plant DataFrames merged successfully at {df_save_path}.")
        
        meta_df = pd.DataFrame()
        for p in meta_df_paths:
            meta_df = meta_df.append(pd.read_pickle(p), ignore_index=True)

        df_save_path = os.path.join(self.save_dir, "meta.pkl")
        meta_df.to_pickle(df_save_path)
        logger.info(f"{self.name} -> Metadata DataFrames merged successfully at {df_save_path}.")

        quality_df = pd.DataFrame()
        for p in quality_df_paths:
            quality_df = quality_df.append(pd.read_pickle(p), ignore_index=True)

        df_save_path = os.path.join(self.save_dir, "det_quality.pkl")
        quality_df.to_pickle(df_save_path)
        logger.info(f"{self.name} -> Detection Quality DataFrames merged successfully at {df_save_path}.")
        
        self.save(obj="", name="_dummy", type_="json")


class MakeImageDataset(Task):
    
    def __init__(
            self,
            df_dir: str,
            source_tiff_dir: str,
            source_channels: List[str],
            export_channels: List[str],
            export_shape: List[int],
            export_resolution: float,
            nan_value: Union[str, float, int],
            ann_df_path: str,
            ann_gps_name: str,
            ann_values_name: str,
            tol_distance: float,
            save_dir: str
        ):
        super().__init__()
        self.df_dir = df_dir
        self.source_tiff_dir = source_tiff_dir
        self.source_channels = np.asarray(source_channels)
        self.export_channels = np.asarray(export_channels)
        self.export_shape = export_shape
        self.export_resolution = export_resolution
        if type(nan_value) == str:
            if nan_value == "nan":
                self.nan_value = np.nan
            else:
                raise AttributeError()
        else:
            self.nan_value = nan_value
        self.ann_df_path = ann_df_path
        self.ann_gps_name = ann_gps_name
        self.ann_values_name = ann_values_name
        self.tol_distance = tol_distance
        self.save_dir = save_dir

    @staticmethod
    def point_inside_region(
            point: Tuple,
            region
        ) -> bool:
        r, c = point
        minr, minc, maxr, maxc = region.bbox
        if (minr <= r <= maxr) and (minc <= c <= maxc):
            return True
        else:
            return False

    def find_bbox(
            self,
            img: np.ndarray, 
            segmask: Optional[np.ndarray]
        ) -> Optional[Tuple[Tuple, np.ndarray]]:
        if segmask is not None:
            distance = distance_transform_edt(segmask)
            coords = peak_local_max(distance, min_distance=int(self.mean_shape/10), exclude_border=False)
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers = label(mask)
            plant_labels = watershed(-self.kernel, markers, mask=segmask)
            plant_regions = regionprops(plant_labels)
        else:
            plant_labels = np.ones(img.shape[:2], dtype=int)
        
        plant_regions = np.asarray(regionprops(plant_labels, intensity_image=self.kernel))

        valid = np.array([self.point_inside_region((self.export_shape[0]//2, self.export_shape[1]//2), r) for r in plant_regions], dtype=bool)
        if np.sum(valid) > 0:

            if np.sum(valid) > 1:
                v = np.argmax([plant_regions[i].area if valid[i] else 0 for i in range(len(plant_regions))])
            else:
                v = np.argmax(valid)

            region = plant_regions[v]
            minr, minc, maxr, maxc = region.bbox
            if segmask is not None:
                retmask = segmask.copy()
                retmask[:minr,:] = 0
                retmask[maxr:,:] = 0
                retmask[:,:minc] = 0
                retmask[:,maxc:] = 0
            else:
                retmask = plant_labels
            
            return (minc, minr, maxc, maxr), retmask
        
        else:
            return None

    def retrieve_annotations(
            self,
            plants: pd.DataFrame,
            meta: pd.DataFrame
        ) -> pd.DataFrame:
        nn = NearestNeighbors(n_neighbors=1)
        ann = pd.read_pickle(self.ann_df_path)
        plants_ann = []
        for d in pd.DatetimeIndex(np.unique(ann.date)):
            utm = meta[(meta.date == d) & (meta.field_id == self.field_id)].iloc[0].utm_transform
            ann_ = ann[(ann.date == d) & (ann.field_id == self.field_id)]
            
            if len(ann_) > 0:
                plants_ = plants[plants.date == d].copy()
                utm_ann = np.asarray(utm(*np.stack(ann_[self.ann_gps_name].values).T)).T
                utm_plants = np.asarray(utm(*np.stack(plants_[self.ann_gps_name].values).T)).T
                nn.fit(utm_ann)
                dist, ind = nn.kneighbors(utm_plants)
                distmask = dist.flatten() < self.tol_distance/100.
                plants_idx = plants_[distmask].index
                plants_[self.ann_values_name] = None
                for i, v in zip(plants_idx, ann_[self.ann_values_name].iloc[ind.flatten()[distmask]]):
                    plants_.loc[i, self.ann_values_name] = v
                plants_ = plants_[plants_[self.ann_values_name].notna()]
                plants_ann.append(plants_)
            else:
                logger.warn(f"{self.name}-{self.field_id}-{d.date()} -> no annotations found.")

        return pd.concat(plants_ann)
    
    @staticmethod
    def plot(
            ax,
            image,
            mask,
            bbox
        ):
        ax.imshow(image/image.max(), alpha=1)
        ax.imshow(mask, vmin=0, vmax=1, alpha=0.2, cmap="gray")
        min_row, min_col, max_row, max_col = bbox
        xy = (min_col, min_row)
        w = max_col-min_col
        h = max_row-min_row
        b = mpl.patches.Rectangle(xy, w, h, fill=False, edgecolor="r")
        ax.add_patch(b)
        
        return ax

    def run(
            self,
            field_id: str,
            dates: pd.DatetimeIndex,
            segmentation_masks: List[np.ndarray],
            px_resolutions: np.ndarray,
        ):
        temp_dir = os.path.join(self.save_dir, "temp")

        # make directory if not existing
        makeDirectory(temp_dir)

        self.field_id = field_id

        # some precalculations
        src_channels = [np.argmax(self.source_channels == c)+1 for c in self.export_channels]
        self.mean_shape = np.mean(self.export_shape)
        self.kernel = np.zeros(self.export_shape)
        self.kernel[self.kernel.shape[0]//2, self.kernel.shape[1]//2] = 1
        self.kernel = gaussian(self.kernel, 0.05*self.mean_shape)

        # save dataset info dict
        info_dict = {"channels": self.export_channels.tolist(),
                     "image_shape": self.export_shape,
                     "image_resolution": self.export_resolution}
        with open(os.path.join(self.save_dir, "info.json"), "w") as f:
            json.dump(info_dict, f, indent=4, sort_keys=False)
        
        # load plants information
        plants = pd.read_pickle(os.path.join(self.df_dir, f"{self.field_id}_plants.pkl"))
        
        # process annotations if available
        if self.ann_df_path != "":
            meta = pd.read_pickle(os.path.join(self.df_dir, f"{self.field_id}_meta.pkl"))
            annotations = self.retrieve_annotations(plants, meta)
            annotation_dir = os.path.join(self.save_dir, "annotations")
            makeDirectory(annotation_dir)
            annotation_path = os.path.join(annotation_dir, f"{self.field_id}_{self.ann_values_name}.pkl")
            annotations.to_pickle(annotation_path)
            logger.info(f"{self.name}-{self.field_id} -> saved annotations at {annotation_path}.")

        # list of lengths (for later HDP file generation)
        lengths = []

        # iterate source images and save plant images, segmentation masks, and central bounding boxs        
        for d, segmask, px_res in zip(dates, segmentation_masks, px_resolutions):

            scale_factor = px_res / self.export_resolution

            logger.info(f"{self.name}-{self.field_id}-{d.date()} -> extract images.")
            
            tiff_path = glob(os.path.join(self.source_tiff_dir, f"{self.field_id}_{d.year}{d.month:02g}{d.day:02g}*.tif"))[0]
            P = plants[(plants['field_id']==self.field_id) & (plants['date']==d)]

            with rio.open(tiff_path) as src:

                p_imgs, p_ids = [], []

                for i, p in P.iterrows():
                    w = Window(
                        p.xy_px[1]-self.export_shape[1]/(2*scale_factor),
                        p.xy_px[0]-self.export_shape[0]/(2*scale_factor),
                        self.export_shape[1]/scale_factor,
                        self.export_shape[0]/scale_factor)
                    transform = src.window_transform(w)

                    # Create a new cropped raster to write to
                    profile = src.profile
                    profile.update({
                        'height': self.export_shape[0],
                        'width': self.export_shape[1],
                        'transform': transform,
                        'count': len(self.export_channels)})

                    # read single plant image and rescale with scale factor
                    src_w = src.read(
                        src_channels,
                        out_shape=(len(src_channels), self.export_shape[0], self.export_shape[1]),
                        resampling=Resampling.bilinear,
                        window=w,
                        fill_value=self.nan_value)

                    # if single image has correct shape ...
                    if list(src_w.shape[1:]) == self.export_shape:

                        p_imgs.append(src_w)
                        p_ids.append(p.group_id)

                        # if correctly shaped segmentation mask is existing ...
                        #if isinstance(segmask, np.ndarray):
                        #    w = w.round_offsets()
                        #    wmask = segmask[w.row_off:w.row_off+int(np.round(scale_factor*self.export_shape[0])),
                        #                    w.col_off:w.col_off+int(np.round(scale_factor*self.export_shape[1]))]
                        #    if list(wmask.shape) != [int(np.round(scale_factor*self.export_shape[0])), int(np.round(scale_factor*self.export_shape[1]))]:
                        #        wmask = None  
                        #    else:
                        #        wmask = resize(wmask, self.export_shape).astype(bool) 
                        #else:
                        #    wmask = None
                        # process for bounding box detection
                        #res = self.find_bbox(np.transpose(src_w[:3], axes=(1,2,0)), wmask)
                        #if res is not None:
                        #    bbox, bmask = res
                        
                lengths.append(len(p_ids))
                file_id = f"{self.field_id}_{p.date.strftime('%Y%m%d')}"
                # write images, plant ids, date and field_id to temporary .pkl file
                with open(os.path.join(temp_dir, file_id+".pkl"), "wb") as pkl_file:
                    pickle.dump((p_imgs, p_ids, p.date, self.field_id, len(p_ids)), pkl_file)
                            #fig, ax = plt.subplots()
                            #ax = self.plot(ax, np.transpose(src_w[:3], axes=(1,2,0)), bmask, bbox)
                            #fig.savefig(os.path.join(plot_dir, file_id+".png"))
                            #del fig, ax
                        #else:
                        #    logger.warn(f"{self.name}-{self.field_id}-{p.group_id}-{p.date.strftime('%Y-%m-%d')} -> no plant detected in image center. Skip.")

        logger.info(f"{self.name}-{self.field_id} -> all valid images saved at {temp_dir}.")
        
        self.save(obj=self.field_id, name="field_id", type_="json")
        self.save(obj=lengths, name="lengths", type_="pickle")


class MergeImageFiles(Task):

    def __init__(
            self,
            save_dir: str,
            channels: List[str],
            img_shape: List[int],
            ann_values_name: str,
            planting_date: str
        ):
        super().__init__()
        self.save_dir = save_dir
        self.channels = np.asarray(channels)
        self.img_shape = img_shape
        self.ann_values_name = ann_values_name
        self.planting_date = dt.datetime.strptime(planting_date, "%Y-%m-%d").date()

    def write_to_file(
            self,
            hf,
            i,
            pkl_path,
            ann,
            ann_dates
        ):
        with open(pkl_path, "rb") as pkl_file:
            (p_imgs, p_ids, date, field_id, n) = pickle.load(pkl_file)
        try:
            hf["image"][i:i+n,...] = p_imgs
            #hf["seg_mask"][i:i+n,...] = seg_mask
            #hf["bbox"][i:i+n,...] = bbox
            hf["age"][i:i+n,...] = [(date.date() - self.planting_date).days]*n
            hf["field_id"][i:i+n,...] = [field_id]*n
            hf["plant_id"][i:i+n,...] = p_ids
            hf["date"][i:i+n,...] = [date._date_repr]*n
            if isinstance(ann, pd.DataFrame):
                nearest_ann_date = abs(date-ann_dates).argmin()
                ann_ = ann.query(f"date == '{ann_dates[nearest_ann_date]}' & field_id == '{field_id}'")
                for j, id in enumerate(p_ids):
                    ann__ = ann_[ann_.group_id == id]
                    if len(ann__) == 1:
                        hf[self.ann_values_name][i+j,...] = ann__[self.ann_values_name]
                    elif len(ann__) > 1:
                        logger.warn(f"{self.name} -> ambiguous labels. Skip.")
                        hf[self.ann_values_name][i+j,...] = 0
                    else:
                        hf[self.ann_values_name][i+j,...] = 0
        except Exception as e:
           logger.error(f"{self.name} -> failed to save data of file {pkl_path}. Error: {e}")
        return n

    def run(
            self,
            reduced_results: List[Dict[str, Dict]]
        ):
        files = sorted(glob(os.path.join(self.save_dir, "temp/*.pkl")))
        h5_path = os.path.join(self.save_dir, "dataset.h5")

        # sum number of total images for fixed-sized HDF5 file
        length = np.sum(np.hstack([r["result"]["lengths"] for r in reduced_results]))
        logger.info(f"{self.name} -> merge {length} images into HDF5 file.")

        # make HDF5 file and datasets
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("image", (length, len(self.channels), *self.img_shape), dtype=np.float32)
            #hf.create_dataset("seg_mask", (length, *self.img_shape), dtype=bool)
            #hf.create_dataset("bbox", (length, 4), dtype=np.uint8)
            hf.create_dataset("age", (length,), dtype=np.uint8)                
            str_dt = h5py.special_dtype(vlen=str)
            hf.create_dataset("date", (length,), dtype=str_dt)
            hf.create_dataset("field_id", (length,), dtype=str_dt)
            hf.create_dataset("plant_id", (length,), dtype=np.uint16)

            if len(glob(os.path.join(self.save_dir, f"annotations/*.pkl"))) > 0:
                hf.create_dataset(self.ann_values_name, (length,), dtype=np.uint8)
                ann = pd.concat([pd.read_pickle(f) for f in sorted(glob(os.path.join(self.save_dir, f"annotations/*_{self.ann_values_name}.pkl")))])
                ann_dates = pd.DatetimeIndex(ann.date.unique()).sort_values()
            else:
                ann = None
                ann_dates = None

            logger.info(f"{self.name} -> save to HDF5 file")

            # iteratively add data from temp. pickle files
            start_ind = 0
            for i, pkl_path in enumerate(files):
                file_len = self.write_to_file(hf, start_ind, pkl_path, ann, ann_dates)
                logger.info(f"{self.name} -> {i+1}/{len(files)} files merged to HDF5 file.")
                start_ind += file_len

        logger.info(f"{self.name} -> all images saved to HDF5 file.")

        logger.info(f"{self.name} -> delete temporary files.")
        for file in files:
            os.remove(file)
        os.rmdir(os.path.join(self.save_dir, "temp"))

        self.save(obj="", name="_dummy", type_="json")


class ExportPlantPositions(Task):

    def __init__(
            self,
            df_dir: str):
        super().__init__()
        self.df_dir = df_dir

    def run(
            self,
            field_id: str
        ):
        plants = pd.read_pickle(os.path.join(self.df_dir, f"{field_id}_plants.pkl"))

        kml_outpath = os.path.join(self.df_dir, f"{field_id}_plants.kml")

        kml = simplekml.Kml()

        plants_pts = kml.newfolder(name=f"{field_id} - Plants")

        plants_schema = kml.newschema(name="Plants")
        plants_schema.newsimplefield(name="field_id", type="string", displayname="Field ID")
        plants_schema.newsimplefield(name="plant_id", type="uint", displayname="Plant ID")
        plants_schema.newgxsimplearrayfield(name="detection_dates", type="uint", displayname="Dates of detection")

        croplines_lines = kml.newfolder(name=f"{field_id} - Seeding lines")

        croplines_schema = kml.newschema(name="Seeding Lines")
        croplines_schema.newsimplefield(name="field_id", type="string", displayname="Field ID")
        croplines_schema.newsimplefield(name="seeding_line_id", type="uint", displayname="Seeding line ID")
        croplines_schema.newsimplefield(name="plants_on_seeding_line", type="uint", displayname="Plants on seeding line")

        logger.info(f"{self.name} -> Write plants and seeding line positions into KML file.")

        # plant positions
        for g_id in np.unique(plants[plants.field_id == field_id].group_id):
            group = plants[(plants.field_id == field_id) & (plants.group_id == g_id)]
            lon, lat = group[group.detected == True].lonlat.mean(axis=0)
            p = plants_pts.newpoint(coords=[(lon, lat)])
            p.extendeddata.schemadata.schemaurl = plants_schema.id
            p.extendeddata.schemadata.newsimpledata("field_id", field_id)
            p.extendeddata.schemadata.newsimpledata("plant_id", g_id)
            dates = set(group[group.detected == True].date)
            p.extendeddata.schemadata.newgxsimplearraydata("detection_dates", [d.date() for d in dates])

        # croplines
        for cr_id in np.unique(plants[plants.field_id == field_id].group_cropline_id):
            cropline = plants[(plants.field_id == field_id) & (plants.group_cropline_id == cr_id)]
            lons, lats = np.stack(cropline[cropline.detected == True].groupby("group_id").lonlat.apply(lambda x: np.stack(x).mean(axis=0))).T
            line = croplines_lines.newlinestring(coords=[(lons[0], lats[0]), (lons[-1], lats[-1])])
            line.extendeddata.schemadata.schemaurl = croplines_schema.id
            line.extendeddata.schemadata.newsimpledata("field_id", field_id)
            line.extendeddata.schemadata.newsimpledata("seeding_line_id", cr_id)
            line.extendeddata.schemadata.newsimpledata("plants_on_seeding_line", len(lons))

        kml.save(kml_outpath)

        logger.info(f"{self.name} -> Plant and seeding lines positions exported as KML file.")

        self.save(obj="", name="_dummy", type_="json")
