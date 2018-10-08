import geopandas as gpd
import os


def main():
    gdf = gpd.read_file(os.path.join('GIS', 'output_shapefiles', 'well_picks.shp'))
    gdf = gdf[['API_10', 'GAMx', 'GAMy', 'SURFLON']]
    print(gdf.head())


if __name__ == '__main__':
    main()
