#-*-coding:UTF-8-*-
"""
Resample of of spectral values

Hypersion data to pretrain autoencoder network should have the similiar spectral bands
with data set to be classified.

"""
# Authors: Jingge Xiao <jingge.xiao@gmail.com>
#
# Created on Fri Jul 28 10:21:50 2017

import numpy as np
import gdal
import os
import csv
import scipy.interpolate as sip


# set path of input image data
str_input_path = r'D:\DeepLearning\Exp\data\tif'

# set path of output interpolated image data
str_out_interp_path = r'E:\DeepLearning\Exp\data\tif_interp'

# Paths of files containing spectral information for spectral resampling
path_interp_spec = r'E:\DeepLearning\Exp\data\Unlabled\pos_3d_180.csv'
path_hyper_spec = r'E:\DeepLearning\Exp\data\Unlabled\Hyperion.csv'

# Get spectral information, positions that have spectral values
list_interp_points = []
reader_ip_csv = csv.reader(open(path_interp_spec, encoding='utf-8'))
for i_row in reader_ip_csv:
    list_interp_points.append(float(i_row[0]))
array_interp_points=np.array(list_interp_points)

interp_bands = len(array_interp_points)

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
            
            # Unfold array into pandas DataFrame
            rows = dsmatrix.shape[1]
            cols = dsmatrix.shape[2]
                
            dsmatrix_interp = np.zeros((interp_bands, rows, cols), dtype = np.float)
            
            
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
            outDataset = driver.Create(str_out_tif, cols, rows, interp_bands, gdal.GDT_Float64)
            
            # Define the projection coordinate system
            outDataset.SetGeoTransform(geo_trans_list)
            outDataset.SetProjection(proj_str)
            for i_ob in range(0,interp_bands):
                outDataset.GetRasterBand(i_ob+1).WriteArray(dsmatrix_interp[i_ob,:,:])
            outDataset = None
