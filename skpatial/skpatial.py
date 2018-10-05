__author__ = 'rosskush'
import geopandas as gpd
import rasterio
import numpy as np


class interp2d():
    def __init__(self,gdf,attribute,res=None, ulc=None, lrc=None):
        """

        :param gdf: geopandas GeoDataFrame object, with geometry attribute
        :param attribute: column name in GeoDataFrame object used for spatial interpolation
        :param res: delx and dely representing pixle resolution
        :param ulc: upper left corner (x,y)
        :param lrc: lower right corner (x,y)
        """
        self.gdf = gdf
        self.atr = attribute

        if np.isfinite(ulc[0]) and np.isfinite(lrc[0]):
            self.xmax = lrc[0]
            self.xmin = ulc[0]
            self.ymax = ulc[1]
            self.ymin = lrc[1]
        else:
            self.xmax = gdf.geometry.x.max()
            self.xmin = gdf.geometry.x.min()
            self.ymax = gdf.geometry.y.max()
            self.ymin = gdf.geometry.y.min()

        if np.isfinite(res):
            self.res = res
        else:
            # if res not passed, then res will be the distance between xmin and xmax / 100
            self.res = (self.xmax - self.xmin) / 100
        self.
        self.

    # def points_to_grid(x, y, z, delx, dely):
    def points_to_grid(self):

        ncol = np.ceil((x.max() - x.min()) / delx)
        nrow = np.ceil((y.max() - y.min()) / dely)
        zi, yi, xi = np.histogram2d(y, x, bins=(int(nrow), int(ncol)), weights=z, normed=False)
        counts, _, _ = np.histogram2d(y, x, bins=(int(nrow), int(ncol)))
        zi = zi / counts
        zi = np.ma.masked_invalid(zi)
        array = np.flipud(np.array(zi))
        return array