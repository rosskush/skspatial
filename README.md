
# skspatial
Simple functions for geospatial interpolation using sklearn's KNN machine learning algorithm, simple scipy interpolation routines ie. "linear", or "cubic" and now pykrige for kriging functions. 

Simply load a projected point shapefile with geopandas as a GeoDataFrame, and use skspatial to create interpolated rasters and countor shapefiles that you can bring into your favorite mapping application such as QGIS.  

# Currently in development by:

![Alt text](docs/intera-logo-sm.png?raw=true "Title")

Written by Ross Kushnereit, Intera Geoscience & Engineering Solutions:
Austin, TX
https://www.intera.com/



## Installation

`skspatial` supports Python 3.6

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


## Refrences
http://chris35wills.github.io/gridding_data/

https://timogrossenbacher.ch/2018/03/categorical-spatial-interpolation-with-r/

http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor
