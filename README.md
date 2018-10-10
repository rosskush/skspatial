
# skspatial
Simple functions for geospatial interpolation using sklearn's KNN machine learning algorithm. 

Simply load a projected point shapefile with geopandas as a GeoDataFrame, and use skspatial to create interpolated rasters and countor shapefiles that you can bring into you favorite mapping application such as QGIS.  

# Currently in development by:

![Alt text](docs/intera-logo-sm.png?raw=true "Title")

Written by Ross Kushnereit, Intera Geoscience & Engineering Solutions:
Austin, TX
https://www.intera.com/



## Installation

`skspatial` supports Python 3 but feel free to try it in python 2 if you don't want to let the past go

```bash
    $ git clone https://github.com/rosskush/skspatial.git
    $ cd skspatial
    $ python setup.py install
```

# Reqiuerments

skspatial in its current state reqiures at a minimum 

geopandas

rasterio

and sklearn

If you would like to use the "write_contours" function, which exports contours to a shapefile, then you will need to install flopy as well.

https://github.com/modflowpy/flopy

## Refrences
http://chris35wills.github.io/gridding_data/

https://timogrossenbacher.ch/2018/03/categorical-spatial-interpolation-with-r/

http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor
