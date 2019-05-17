__author__ = 'rosskush'

import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
try:
    import flopy
except:
    pass

class interp2d():
    def __init__(self,gdf,attribute,res=None, ulc=(np.nan,np.nan), lrc=(np.nan,np.nan)):
        """

        :param gdf: geopandas GeoDataFrame object, with geometry attribute
        :param attribute: column name in GeoDataFrame object used for spatial interpolation
        :param res: delx and dely representing pixle resolution
        :param ulc: upper left corner (x,y)
        :param lrc: lower right corner (x,y)
        """

        self.gdf = gdf
        self.attribute = attribute
        self.x = gdf.geometry.x.values
        self.y = gdf.geometry.y.values
        self.z = gdf[attribute].values
        self.crs = gdf.crs

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
        self.extent = (self.xmin, self.xmax, self.ymin, self.ymax)
        if np.isfinite(res):
            self.res = res
        else:
            # if res not passed, then res will be the distance between xmin and xmax / 1000
            self.res = (self.xmax - self.xmin) / 1000

        self.ncol = int(np.ceil((self.xmax - self.xmin) / self.res)) # delx
        self.nrow = int(np.ceil((self.ymax - self.ymin) / self.res))# dely
    # def points_to_grid(x, y, z, delx, dely):
    def points_to_grid(self):
        """

        :return: array of size nrow, ncol

        http://chris35wills.github.io/gridding_data/
        """
        hrange = ((self.ymin,self.ymax),(self.xmin,self.xmax)) # any points outside of this will be condisdered outliers and not used

        zi, yi, xi = np.histogram2d(self.y, self.x, bins=(int(self.nrow), int(self.ncol)), weights=self.z, normed=False,range=hrange)
        counts, _, _ = np.histogram2d(self.y, self.x, bins=(int(self.nrow), int(self.ncol)),range=hrange)
        np.seterr(divide='ignore',invalid='ignore') # we're dividing by zero but it's no big deal
        zi = np.divide(zi,counts)
        np.seterr(divide=None,invalid=None) # we'll set it back now
        zi = np.ma.masked_invalid(zi)
        array = np.flipud(np.array(zi))
    
        return array

    def knn_2D(self, k=15, weights='uniform', algorithm='brute', p=2, maxrows = 1000):

        if len(self.gdf) > maxrows:
            raise ValueError('GeoDataFrame should not be larger than 1000 rows, knn is a slow algorithim and can be too much for your computer, Change maxrows at own risk')  # shorthand for 'raise ValueError()'

        array = self.points_to_grid()
        X = []
        # nrow, ncol = array.shape
        frow, fcol = np.where(np.isfinite(array)) # find areas where finite values exist
        for i in range(len(frow)):
            X.append([frow[i], fcol[i]])
        y = array[frow, fcol]

        train_X, test_x, train_y, test_y = train_test_split(X, y, test_size=1, random_state=123)

        knn = KNeighborsRegressor(n_neighbors=k, weights=weights, algorithm=algorithm, p=2)
        knn.fit(train_X, train_y)
        # print(f'score = {knn.score(train_X,train_y)}')

        X_pred = []
        for r in range(int(self.nrow)):
            for c in range(int(self.ncol)):
                X_pred.append([r, c])
        y_pred = knn.predict(X_pred)
        karray = np.zeros((self.nrow, self.ncol))
        i = 0
        for r in range(int(self.nrow)):
            for c in range(int(self.ncol)):
                karray[r, c] = y_pred[i]
                i += 1
        return karray


    def write_raster(self,array,path):
        if '.' not in path[-4:]:
            path+='.tif'

        # transform = from_origin(gamx.min(), gamy.max(), res, res)
        transform = from_origin(self.xmin, self.ymax, self.res, self.res)


        new_dataset = rasterio.open(path, 'w', driver='GTiff',
                                    height=array.shape[0], width=array.shape[1], count=1, dtype=array.dtype,
                                    crs=self.gdf.crs, transform=transform, nodata=np.nan)
        new_dataset.write(array, 1)
        new_dataset.close()

    def write_contours(self,array,path,base=0,interval=100):
        levels = np.arange(base,array.max(),interval)
        # matplotlib contour objects are shifted half a cell to the left and up
        cextent = np.array(self.extent)
        cextent[0] = cextent[0] + self.res/2.7007
        cextent[1] = cextent[1] + self.res/2.7007
        cextent[2] = cextent[2] - self.res/3.42923
        cextent[3] = cextent[3] - self.res/3.42923


        cs = plt.contour(np.flipud(array),extent=cextent,levels=levels)
        delr = np.ones(int(self.ncol)) * self.res
        delc = np.ones(int(self.nrow)) * self.res
        # print(self.crs)
        sr = flopy.utils.SpatialReference(delr,delc,3,self.xmin,self.ymax)#epsg=self.epsg)
        sr.export_contours(path,cs,epsg=self.crs) #crs_wkt=
        plt.close('all')

    def plot_image(self,array,title=''):
        fig, axes = plt.subplots(figsize=(10,8))
        plt.imshow(array, cmap='jet',extent=self.extent)
        plt.colorbar()
        plt.title(title)
        fig.tight_layout()
        return axes



