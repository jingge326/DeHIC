#-*-coding:UTF-8-*-
"""
Processing unlabeled Hyperion data.

Splitting original Hyperion data into patches for subsequent network training

"""
# Authors: Jingge Xiao <jingge.xiao@gmail.com>
#
# Created on Fri Jul 28 10:21:50 2017

import numpy as np
import gdal
import os
import csv
import gc
import scipy.interpolate as sip


# set path of input image data
str_input_path = r'M:\DeepLearning\Exp\data\tif'

# set path of output interpolated image data
str_out_interp_path = r'M:\DeepLearning\Exp\data\tif_interp'

# Paths of files containing spectral information for spectral resampling
path_interp_spec = r'M:\DeepLearning\Exp\data\Unlabled\pos_3d_180.csv'
path_hyper_spec = r'M:\DeepLearning\Exp\data\Unlabled\Hyperion.csv'

# set path of output scaled image data
str_out_scaled_path = r'M:\DeepLearning\Exp\data\tif_scaled_interp'

# set output path of statistics data file
str_stat_path = r"M:\DeepLearning\Exp\data\tif_scaled_interp\stat.npy"


# Get spectral information, positions that have spectral values
list_interp_points = []
reader_ip_csv = csv.reader(open(path_interp_spec, encoding='utf-8'))
for i_row in reader_ip_csv:
    list_interp_points.append(float(i_row[0]))
array_interp_points=np.array(list_interp_points)

list_hyper_points = []
reader_hyper_csv = csv.reader(open(path_hyper_spec, encoding='utf-8'))
for i_row in reader_hyper_csv:
    list_hyper_points.append(float(i_row[0]))
array_hyper_points=np.array(list_hyper_points)


for i_files in os.walk(str_input_path):
    for str_name in i_files[2]:
        str_file_format = str_name[-3:]
        if str_file_format == 'tif':
            str_tif_path = os.path.join(i_files[0], str_name)
            
            # Read image
            dataset = gdal.Open(str_tif_path)
            dsmatrix = dataset.ReadAsArray(xoff=0, yoff=0, xsize=dataset.RasterXSize, ysize=dataset.RasterYSize)
            
            # Get Geographic meta data
            geo_trans_list = dataset.GetGeoTransform()
            proj_str = dataset.GetProjection()
            num_bands = dataset.RasterCount
            # Unfold array into pandas DataFrame
            rows = dsmatrix.shape[1]
            cols = dsmatrix.shape[2]
                
            dsmatrix_interp = np.zeros((len(array_interp_points), rows, cols), dtype = np.float)
            
            for i_row in range(0, rows):
                for i_col in range(0, cols):
                    v_pixels = dsmatrix[:, i_row, i_col]
                    if np.mean(v_pixels) != 0:
                        f = sip.interp1d(array_hyper_points, v_pixels)
                        dsmatrix_interp[:, i_row, i_col] = f(array_interp_points)
                    
            # Set output file path
            str_out_tif = os.path.join(str_out_interp_path, str_name)
            
            # Output result in "GeoTiff" format           
            driver=gdal.GetDriverByName("GTiff")
            driver.Register()
            outDataset = driver.Create(str_out_tif, cols, rows, num_bands, gdal.GDT_Float64)
            
            # Define the projection coordinate system
            outDataset.SetGeoTransform(geo_trans_list)
            outDataset.SetProjection(proj_str)
            for i_ob in range(0,num_bands):
                outDataset.GetRasterBand(i_ob+1).WriteArray(dsmatrix_interp[i_ob,:,:])
            outDataset = None
 
    
 # collect memory
del(dsmatrix, dsmatrix_interp)
gc.collect()

       
# read the list of files
list_files_path = []
for i in os.walk(str_out_interp_path):
    for j in i[2]:
        str_file_format = j[-3:]
        if str_file_format == 'tif':
            str_tif_path = os.path.join(i[0], j)
            list_files_path.append(str_tif_path)



# Compute the global mean and the global standard deviation
list_stat = []
for i_band in range(0, 175):
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
                if data_pixel != 0:
                    list_data.append(data_pixel)
    
        print(f)
    
    value_mean = np.mean(list_data)
    
    value_std = np.std(list_data)
    
    list_stat.append((i_band, value_mean, value_std))
    
    print(list_stat[i_band])
    

np.save(str_stat_path, list_stat) 



# Scale data
# read the list of files
for i_files in os.walk(str_out_interp_path):
    for str_name in i_files[2]:
        str_file_format = str_name[-3:]
        if str_file_format == 'tif':
            str_tif_path = os.path.join(i_files[0], str_name)
            
            # Read image
            dataset = gdal.Open(str_tif_path)
            dsmatrix = dataset.ReadAsArray(xoff=0, yoff=0, xsize=dataset.RasterXSize, ysize=dataset.RasterYSize)
            
            # Get Geographic meta data
            geo_trans_list = dataset.GetGeoTransform()
            proj_str = dataset.GetProjection()
            num_bands = dataset.RasterCount
            # Unfold array into pandas DataFrame
            rows = dsmatrix.shape[1]
            cols = dsmatrix.shape[2]
                
            dsmatrix_scaled = np.zeros_like(a = dsmatrix, dtype = np.float)
            
            for i_band in range(0, num_bands):
                for i_row in range(0, rows):
                    for i_col in range(0, cols):
                        data_pixel = dsmatrix[i_band, i_row, i_col]
                        if data_pixel != 0:
                            dsmatrix_scaled[i_band, i_row, i_col] = (data_pixel - list_stat[i_band][1]) / list_stat[i_band][2]
        
            # Set output file path
            str_out_tif = os.path.join(str_out_scaled_path, str_name)
            
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
