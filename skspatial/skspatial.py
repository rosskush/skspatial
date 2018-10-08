__author__ = 'rosskush'

import geopandas as gpd
import rasterio
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

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

        if np.isfinite(res):
            self.res = res
        else:
            # if res not passed, then res will be the distance between xmin and xmax / 100
            self.res = (self.xmax - self.xmin) / 100


    # def points_to_grid(x, y, z, delx, dely):
    def points_to_grid(self):
        """

        :return: array of size nrow, ncol
        """
        ncol = np.ceil((self.xmax - self.xmin) / self.res) # delx
        nrow = np.ceil((self.ymax - slef.ymin) / self.res) # dely
        zi, yi, xi = np.histogram2d(y, x, bins=(int(nrow), int(ncol)), weights=z, normed=False)
        counts, _, _ = np.histogram2d(y, x, bins=(int(nrow), int(ncol)))
        zi = zi / counts
        zi = np.ma.masked_invalid(zi)
        array = np.flipud(np.array(zi))
        return array

    def knn_2D(self, k=15, weights='uniform', algorithm='brute'):
        array = points_to_grid(self)
        X = []
        nrow, ncol = array.shape
        frow, fcol = np.where(np.isfinite(array))
        for i in range(len(frow)):
            X.append([frow[i], fcol[i]])
        y = array[frow, fcol]

        train_X, test_x, train_y, test_y = train_test_split(X, y, test_size=1, random_state=123)

        knn = KNeighborsRegressor(n_neighbors=k, weights=weights, algorithm=algorithm, p=2)
        knn.fit(train_X, train_y)
        # print(f'score = {knn.score(train_X,train_y)}')

        X_pred = []
        for r in range(nrow):
            for c in range(ncol):
                X_pred.append([r, c])
        y_pred = knn.predict(X_pred)
        karray = np.zeros((nrow, ncol))
        i = 0
        for r in range(nrow):
            for c in range(ncol):
                karray[r, c] = y_pred[i]
                i += 1
        return karray


