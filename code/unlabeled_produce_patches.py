#-*-coding:UTF-8-*-
"""
Splitting original Hyperion data into patches for subsequent network training

"""
# Authors: Jingge Xiao <jingge.xiao@gmail.com>
#
# Created on Fri Jul 28 10:21:50 2017

import numpy as np
import gdal
import os
import gc
import matplotlib.pyplot as plt

# Set the dimension of convolution kernel(Only 2 or 3 is supported)
conv_d = 2

# Set patch size, which have to be same as that of labeled data.
img_size = 8

# Set number of bands adopted in experiments
num_bands = 180

# set path of output data
if conv_d == 2:
    path_hyper = r"M:\DeepLearning\Exp\data\ing\180\hyper\2d_conv"
else:
    path_hyper = r"M:\DeepLearning\Exp\data\ing\180\hyper\3d_conv"

path_sub_patches = os.path.join(path_hyper, "list")

path_unlab_patches = os.path.join(path_hyper, "unlabel_patches.npy")

# set path of scaled image data
# each input data is a stack of bands and saved as tif format
path_scaled = r'M:\DeepLearning\Exp\data\ing\180\tif_interp_scaled'

# read the list of files
fileObjectsList = []
for i in os.walk(path_scaled):
    for j in i[2]:
        fileFormatStr = j[-3:]
        if fileFormatStr == 'tif':
            fileObjectsList.append(j)

# get patches from original data
list_sub_image = []
num_list = 1
for f in fileObjectsList:
    print(f)
    imagery_path = os.path.join(path_scaled, f)

    # read tif files
    dataset = gdal.Open(imagery_path)
    dsmatrix = dataset.ReadAsArray(xoff=0, yoff=0, xsize=dataset.RasterXSize, ysize=dataset.RasterYSize)
    image_rows = dsmatrix.shape[1]
    image_cols = dsmatrix.shape[2]    
    image_spec = dsmatrix.shape[0]
    
    # construct pixel patches within original data margins
    for i in range(0, image_rows, img_size):
        if i >= image_rows - img_size:
            break
        for j in range(0, image_cols, img_size):
            if j >= image_cols - img_size:
                break
            break_flag = False
            sub_image = dsmatrix[:, i:i+img_size, j:j+img_size]
            
            # assign pixel values
            for p in range(0, sub_image.shape[1]):
                for q in range(0, sub_image.shape[2]):
                    
                    # patches with "NoData" pixel is not allowed
                    if np.mean(sub_image[:, p, q]) == 0.0:
                        break_flag = True
                        break
                    
                if break_flag == True:
                    break
            
            if break_flag == False:
                if conv_d == 2:
                    list_sub_image.append(sub_image)
                else:
                    list_sub_image.append(sub_image.reshape((1, num_bands, img_size, img_size)))
                
            
            # limit the length of lists in order to avoid much too large npy files
            if(len(list_sub_image)==10000):
                array_sub_image=np.array(list_sub_image)
                
                # output array to file
                np.save(os.path.join(path_sub_patches, str(num_list)+'.npy'), array_sub_image)
                num_list = num_list + 1
                list_sub_image = []

    # collect memory
    del(dsmatrix)
    gc.collect()

# output the last list to file  
array_sub_image=np.array(list_sub_image)
np.save(os.path.join(path_sub_patches, 'last.npy'), array_sub_image)


list_dir_strs = os.listdir(path_sub_patches)
name_str = list_dir_strs[0]
npy_path = os.path.join(path_sub_patches, name_str)
array_patches = np.load(npy_path)
for i_name in np.arange(1, len(list_dir_strs)):
    name_str = list_dir_strs[i_name]
    npy_path = os.path.join(path_sub_patches, name_str)
    tmp_array_patches = np.load(npy_path)
    array_patches = np.vstack((array_patches,tmp_array_patches)) 
np.save(path_unlab_patches, array_patches) 


# Schematic diagram of spectral curve
# This is not necessary for data preprocess, if you want to observe the curve of 
# data points, you can modify and use these lines.
d_y = array_sub_image[300,:,0,0]
d_x = np.arange(0,len(array_sub_image[0,:,0,0]),1)
plt.plot(d_x, d_y)