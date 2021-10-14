from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed, clear_border
from skimage.measure import label, regionprops
from skimage.measure._regionprops import RegionProperties
from skimage.color import label2rgb, rgb2hsv, rgb2luv
from skimage.feature import peak_local_max
from sklearn.mixture import GaussianMixture

from plant_extraction.vi import *

class PlantImage():

    def __init__(self, img: np.ndarray, channels: List[str]):
        assert img.shape[0] == len(channels)
        self.channels = np.asarray(channels)
        self.img = img

    def read_channel(self, channel_name: str) -> np.ndarray:
        assert channel_name in self.channels
        return self.img[np.argmax(self.channels==channel_name),:,:]

    def read_channels(self, channel_names: List[str]) -> np.ndarray:
        return self.img[[np.argmax(self.channels==c) for c in channel_names],:,:]

    def get_image(self, soil_mask: bool = False, vi: str = "GLI"):
        if soil_mask:
            img = np.transpose(self.img.copy(), axes=(1,2,0))
            img[np.invert(self.segmentation_mask(vi))] = np.zeros(self.img.shape[0])
            return np.transpose(img, axes=(2,0,1))
        else:
            return self.img

    def plot_rgb(self, soil_mask: bool = False, vi: str = "GLI") -> Tuple[plt.Figure, plt.axis]:
        img = self.get_image(soil_mask, vi)
        
        rgb = img[[np.argmax(self.channels=="R"),
                   np.argmax(self.channels=="G"),
                   np.argmax(self.channels=="B")],:,:]
        rgb = np.transpose(rgb, axes=(1,2,0))

        fig, ax = plt.subplots()
        ax.imshow(rgb)
        return fig, ax

    def segmentation_mask(self, vi: str = "GLI") -> np.ndarray:
        if vi == "NGRDI":
            vi_image = ngrdiImage(R = self.read_channel("R"), G = self.read_channel("G"))
        elif vi == "GLI":
            vi_image = gliImage(R = self.read_channel("R"), G = self.read_channel("G"), B = self.read_channel("G"))
        elif vi == "OSAVI":
            vi_image = osaviImage(R = self.read_channel("R"), NIR = self.read_channel("NIR"), y_osavi = 0.6)
        else:
            raise NotImplementedError()

        rgb = np.transpose(self.read_channels(["R", "G", "B"]), axes=(1,2,0))
        hsv = rgb2hsv(rgb)
        luv = rgb2luv(rgb)

        all = np.dstack((100*vi_image, np.transpose(self.get_image(), axes=(1,2,0)), hsv, luv))

        gm = GaussianMixture(n_components=2).fit(all.reshape(-1,all.shape[-1]))
        labels = gm.predict(all.reshape(-1,all.shape[-1])).reshape(all.shape[:2],)
        seg_mask = labels == np.argmax(gm.means_[:,0])

        return seg_mask

    def detect_objects(self, vi: str = "GLI", plot: bool = True) -> Tuple[np.ndarray, np.ndarray, List[RegionProperties], List[RegionProperties]]:
        segmask = self.segmentation_mask(vi)
        #cleared = clear_border(segmask)
        distance = ndi.distance_transform_edt(segmask)
        coords = peak_local_max(distance, footprint=np.ones((6, 6)), labels=segmask, threshold_rel=0.3)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers = label(mask)
        #markers, _ = ndi.label(mask)
        plant_labels = watershed(markers, mask=segmask)
        plant_regions = regionprops(plant_labels)

        leaf_labels = watershed(-distance, markers, mask=segmask)
        leaf_regions = regionprops(leaf_labels)

        if plot:
            #label_image = label(segmask)
            #image_label_overlay = label2rgb(label_image, image=self.get_image(), bg_label=0)
            image_label_overlay = label2rgb(leaf_labels, image=np.transpose(self.read_channels(["R", "G", "B"]), axes=(1,2,0)), bg_label=0)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(image_label_overlay)

            #for region in regionprops(label_image):
            #for region in leaf_regions:
            #    # take regions with large enough areas
            #    if region.area >= 16:
            #        # draw rectangle around segmented leaves
            #        minr, minc, maxr, maxc = region.bbox
            #        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
            #                                fill=False, edgecolor='gold', linewidth=2)
            #        ax.add_patch(rect)

            for region in plant_regions:
                # take regions with large enough areas
                if region.area >= 64:
                    # draw rectangle around segmented leaves
                    minr, minc, maxr, maxc = region.bbox
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                            fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)

            ax.set_axis_off()
            plt.tight_layout()
            plt.show()

        return plant_labels, leaf_labels, plant_regions, leaf_regions


class PlantImageStack():

    def __init__(self, channels: List[str]):
        self.channels = channels

    def load_numpy(self, stack: np.ndarray):
        self.stack = []

        for i in stack:
            self.stack.append(PlantImage(img=i, channels=self.channels))
        
        return self

    def __len__(self):
        return len(self.stack)
