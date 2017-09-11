#-*-coding:UTF-8-*-
"""
Normalizing pixel values within each band

Mean and std are calculated using all unlabelled data

"""
# Authors: Jingge Xiao <jingge.xiao@gmail.com>
#
# Created on Fri Jul 28 10:21:50 2017

import numpy as np
import gdal
import os
import gc

interp_bands = 180

# set path of interpolated image data
path_interp = r'E:\DeepLearning\Exp\data\tif_interp'

# set output path of statistics data file
path_stat = r"E:\DeepLearning\Exp\data\tif_scaled_interp\stat.npy"

# set path of scaled image data
path_scaled = r'E:\DeepLearning\Exp\data\tif_scaled_interp'

# read the list of files
list_files_path = []
for i in os.walk(path_interp):
    for j in i[2]:
        str_file_format = j[-3:]
        if str_file_format == 'tif':
            str_tif_path = os.path.join(i[0], j)
            list_files_path.append(str_tif_path)



# Compute the global mean and the global standard deviation
list_stat = []
for i_band in range(0, interp_bands):
    list_data = []
    for f in list_files_path:
        # Read image
        dataset = gdal.Open(f)
        dsmatrix = dataset.ReadAsArray(xoff=0, yoff=0, xsize=dataset.RasterXSize, ysize=dataset.RasterYSize)
        rows = dsmatrix.shape[1]
        cols = dsmatrix.shape[2]
        for i_row in range(0, rows):
            for i_col in range(0, cols):
                data_pixel = dsmatrix[i_band, i_row, i_col]
                if data_pixel > 0.0:
                    list_data.append(data_pixel)
    
        print(f)
    
    value_mean = np.mean(list_data)
    
    value_std = np.std(list_data)
    
    list_stat.append((i_band, value_mean, value_std))
    
    print(list_stat[i_band])
    

np.save(path_stat, list_stat) 

 # collect memory
del(dsmatrix)
gc.collect()

array_stat = np.load(path_stat)

# Scale data
# read the list of files
for i_files in os.walk(path_interp):
    for str_name in i_files[2]:
        str_file_format = str_name[-3:]
        if str_file_format == 'tif':
            str_tif_path = os.path.join(i_files[0], str_name)
            
            # Read image
            dataset = gdal.Open(str_tif_path)
            dsmatrix_scaled = dataset.ReadAsArray(xoff=0, yoff=0, xsize=dataset.RasterXSize, ysize=dataset.RasterYSize)
            
            # Get Geographic meta data
            geo_trans_list = dataset.GetGeoTransform()
            proj_str = dataset.GetProjection()
            num_bands = dataset.RasterCount
            # Unfold array into pandas DataFrame
            rows = dsmatrix_scaled.shape[1]
            cols = dsmatrix_scaled.shape[2]
            
            
            for i_row in range(0, rows):
                for i_col in range(0, cols):
                    if np.mean(dsmatrix_scaled[:, i_row, i_col])!=0.0:
                        for i_band in range(0, num_bands):
                            data_pixel = dsmatrix_scaled[i_band, i_row, i_col]
                            dsmatrix_scaled[i_band, i_row, i_col] = (data_pixel - array_stat[i_band, 1]) / array_stat[i_band, 2]
            
            # Set output file path
            str_out_tif = os.path.join(path_scaled, str_name)
            
            # Output result in "GeoTiff" format           
            driver=gdal.GetDriverByName("GTiff")
            driver.Register()
            outDataset = driver.Create(str_out_tif, cols, rows, num_bands, gdal.GDT_Float64)
            
            # Define the projection coordinate system
            outDataset.SetGeoTransform(geo_trans_list)
            outDataset.SetProjection(proj_str)
            for i_ob in range(0,num_bands):
                outDataset.GetRasterBand(i_ob+1).WriteArray(dsmatrix_scaled[i_ob,:,:])
            outDataset = None
