import rasterio
import geopandas as gp
from shapely.geometry import Point


def extract_raster(raster_path,xy):
    # raster_path : path to raster
    # xy: list or array of tuples of x,y i.e [(x1,y1),(x2,y2)...(xn,yn)]
    # returns list of length xy or sampled values
    raster = rasterio.open(raster_path)
    nodata = raster.nodata

    values = []
    for i, xyi in enumerate(xy):
        try:
            value = list(raster.sample([xyi]))[0][0]
        except:
            value = np.nan

        if value == nodata: # if value is nodata from rasdter, set to nan
            value = np.nan

        values.append(value)

    # values = [item[0] for item in values] # list comprehension to get the value
    return values

def raster2pts(rasobj, column='Value'):
    crs = rasobj.crs
    array = rasobj.read(1)

    nrow, ncol = rasobj.height, rasobj.width

    data = []
    i = 0
    for r in range(nrow):
        for c in range(ncol):

            x,y = rasobj.xy(r,c)
            z = array[r,c]
            data.append([z,Point(x,y)])

            i+=1

    gdf = gpd.GeoDataFrame(data,columns=[column,'geometry'],crs=crs)

    return gdf
