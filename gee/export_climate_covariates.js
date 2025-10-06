// GEE: Export monthly Precip (CHIRPS) & Temp (ERA5-Land) aligned with NDVI
var region = ee.FeatureCollection('users/yourname/study_area');
var start = '2015-01-01';
var end = '2024-12-31';
var scale = 500; // MODIS resolution
var bucket = 'your-gcs-bucket';

// Precip: CHIRPS daily → monthly mean
var precip = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate(start, end)
  .select('precipitation')
  .map(img => img.clip(region))
  .toBands()
  .rename(['Precip']);

// Temp: ERA5 hourly → daily mean → monthly
var temp = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
  .filterDate(start, end)
  .select('temperature_2m')
  .map(img => img.subtract(273.15).clip(region))
  .reduce(ee.Reducer.mean())
  .rename('Temp');

Export.image.toCloudStorage({
  image: precip,
  description: 'Precip_monthly',
  bucket: bucket,
  fileNamePrefix: 'climate/Precip',
  scale: scale,
  region: region.geometry(),
  fileFormat: 'NetCDF'
});

Export.image.toCloudStorage({
  image: temp,
  description: 'Temp_monthly',
  bucket: bucket,
  fileNamePrefix: 'climate/Temp',
  scale: scale,
  region: region.geometry(),
  fileFormat: 'NetCDF'
});
