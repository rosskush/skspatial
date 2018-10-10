# skspatial
Simple Functions for spatial interpolation using machine learning

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

If you would like to use the "write_contour" function, which exports contours to a shapefile, then you will need to install flopy as well.

https://github.com/modflowpy/flopy


## Refrences
http://chris35wills.github.io/gridding_data/

https://timogrossenbacher.ch/2018/03/categorical-spatial-interpolation-with-r/

http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor