// GEE: Export monthly Precip (CHIRPS) & Temp (ERA5-Land) aligned with NDVI
// Google Earth Engine: Export Climate Covariates for GW-NDVI Analysis
// Author: Your Name
// Date: 2025-10-06
// GEE 导出气候变量脚本（降水 + 温度）
// 支持 CHIRPS 降水 与 ERA5-Land 温度
// 自动按区域（GeoJSON）裁剪、重采样至 NDVI 分辨率
// 导出为 NetCDF（兼容 xarray）
// 使用说明：
// 将 region 替换为你自己的研究区（上传 GeoJSON 到 GEE Assets）
// 在 Google Cloud Storage 创建 bucket 并授权 GEE 写入权限
// 导出后下载 .nc 文件，用于 Python 脚本
// 
// ========== 用户配置 ==========
var region = ee.FeatureCollection('users/yourname/study_area'); // 替换为你的 GeoJSON 路径
var startDate = '2015-01-01';
var endDate = '2024-12-31';
var ndviResolution = 500; // 单位：米（如 MODIS NDVI）
var exportBucket = 'your-gcs-bucket'; // Google Cloud Storage

// ========== 数据源 ==========
// Precipitation (CHIRPS)
var precip = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate(startDate, endDate)
  .select('precipitation')
  .map(function(img) {
    return img.clip(region).rename('Precip');
  });

// Temperature (ERA5-Land, convert K to °C)
var temp = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
  .filterDate(startDate, endDate)
  .select('temperature_2m')
  .map(function(img) {
    // 转为摄氏度
    return img.subtract(273.15).clip(region).rename('Temp');
  })
  .mean() // 转为日均温（可改为 .max()/.min()）
  .set('system:time_start', ee.Date(startDate).millis());

// 合并为月尺度（与 NDVI 对齐）
var monthlyPrecip = precip
  .filter(ee.Filter.calendarRange(2015, 2024, 'year'))
  .map(function(img) {
    var d = ee.Date(img.get('system:time_start'));
    return img.set('year', d.get('year')).set('month', d.get('month'));
  })
  .reduce(ee.Reducer.mean())
  .rename('Precip');

var monthlyTemp = temp
  .filter(ee.Filter.calendarRange(2015, 2024, 'year'))
  .map(function(img) {
    var d = ee.Date(img.get('system:time_start'));
    return img.set('year', d.get('year')).set('month', d.get('month'));
  })
  .reduce(ee.Reducer.mean())
  .rename('Temp');

// ========== 导出设置 ==========
var exportConfig = {
  scale: ndviResolution,
  region: region.geometry(),
  fileFormat: 'NetCDF',
  fileNamePrefix: 'climate_covariates',
  bucket: exportBucket,
  maxPixels: 1e13
};

Export.image.toCloudStorage({
  image: monthlyPrecip,
  description: 'Precip_monthly',
  ...exportConfig
});

Export.image.toCloudStorage({
  image: monthlyTemp,
  description: 'Temp_monthly',
  ...exportConfig
});
