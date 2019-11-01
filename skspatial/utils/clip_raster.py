import json
import geopandas as gdf
import rasterio

def getFeatures(gdf):
    import json
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""

    return [json.loads(gdf.to_json())['features'][0]['geometry']]
    # return [json.loads(gdf.loc[gdf['geometry'].area == gdf['geometry'].area.max()].to_json())['features'][0]['geometry']]




def clip_raster(rasterobj, clip_obj, path):
    '''
    rasterobj - rasterio.read()
    clip_obj - clip shapefile, make sure there is only 1 geometry
    parth to export clipped raster
    '''
    coords = getFeatures(clip_obj)

    # print(coords)
    out_img, out_transform = mask(dataset=rasterobj, shapes=coords, crop=True)
    # print(out_img)
    out_img = np.array(out_img[0])
    new_dataset = rasterio.open(os.path.join(path), 'w',
                                driver='GTiff', height=out_img.shape[0],
                                width=out_img.shape[1], count=1, dtype=out_img.dtype, crs=clip_obj.crs,
                                transform=out_transform, nodata=np.nan)

    new_dataset.write(out_img, 1)
    new_dataset.close()
    return new_dataset