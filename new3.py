import rasterio

def get_geolocation(mosaic_path):
  with open(mosaic_path) as src:
    # Extract georeference information
    transform = src.transform
    
    latitude_band = src.read(1)  # Assuming band 1 stores latitude
    longitude_band = src.read(2)  # Assuming band 2 stores longitude  
  return transform, latitude_band, longitude_band