import rasterio



def extract_raster(raster_path,xy):
    # raster_path : path to raster
    # xy: list or array of tuples of x,y i.e [(x1,y1),(x2,y2)...(xn,yn)]
    # returns list of length xy or sampled values
    raster = rasterio.open(raster_path)
    values = list(raster.sample(xy)) # convert generator object to list

    values = [item[0] for item in values] # list comprehension to get the value
    return values