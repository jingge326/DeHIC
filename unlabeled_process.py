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
import gc
import matplotlib.pyplot as plt

# set batch size, which have to be same as that of labeled data.
img_size = 15

# set path of output data
npy_str = r'E:\Research\HyperspectralImageClassification\Experiment\Data\npy\hyper'

# set path of input data
# each input data is a stack of bands and saved as tif format
tif_str = r'E:\Research\HyperspectralImageClassification\Experiment\Data\Unlabeled\ing\tif'

# read the list of files
fileObjectsList = []
for i in os.walk(tif_str):
    for j in i[2]:
        fileFormatStr = j[-3:]
        if fileFormatStr == 'tif':
            fileObjectsList.append(j)

# get batches from original data
list_sub_image = []
num_list = 1
for f in fileObjectsList:
    print(f)
    imagery_path = os.path.join(tif_str, f)

    # read tif files
    dataset = gdal.Open(imagery_path)
    dsmatrix = dataset.ReadAsArray(xoff=0, yoff=0, xsize=dataset.RasterXSize, ysize=dataset.RasterYSize)
    image_rows = dsmatrix.shape[1]
    image_cols = dsmatrix.shape[2]    
    image_spec = dsmatrix.shape[0]
    
    # construct pixel batches within original data margins
    for i in range(0, image_rows, img_size):
        if i >= image_rows - img_size:
            break
        for j in range(0, image_cols, img_size):
            if j >= image_cols - img_size:
                break
            break_flag = False
            sub_image = dsmatrix[:, i:i+img_size, j:j+img_size]
            my_sub_image = np.zeros((img_size, img_size, image_spec), dtype = np.int)
            
            # assign pixel values
            for p in range(0, my_sub_image.shape[0]):
                for q in range(0, my_sub_image.shape[1]):
                    my_sub_image[p, q, :] = sub_image[:, p, q]
                    
                    # batches with "NoData" pixel is not allowed
                    if np.mean(my_sub_image[p, q, :]) == 0:
                        break_flag = True
                        break
                    
                if break_flag == True:
                    break
                
            if break_flag == True:
                continue
            
            my_sub_image = my_sub_image.reshape(1, img_size, img_size, my_sub_image.shape[2])
            list_sub_image.append(my_sub_image)
            
            # limit the length of lists in order to avoid much too large npy files
            if(len(list_sub_image)==1000):
                array_sub_image=np.concatenate(list_sub_image, axis=0)
                
                # output array to file
                np.save(os.path.join(npy_str, str(num_list)+'.npy'), array_sub_image)
                num_list = num_list + 1
                list_sub_image = []

    # collect memory
    del(dsmatrix)
    gc.collect()

# output the last list to file  
array_sub_image=np.concatenate(list_sub_image, axis=0)
np.save(os.path.join(npy_str, 'last.npy'), array_sub_image)



# Schematic diagram of spectral curve
# This is not needed for data preprocess, if you want to observe the curve of 
# data points, you can modify and use these lines.
d_y = array_sub_image[300,0,0,:]
d_x = np.arange(0,len(array_sub_image[0,0,0,:]),1)
plt.plot(d_x, d_y)