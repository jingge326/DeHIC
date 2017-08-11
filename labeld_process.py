#-*-coding:UTF-8-*-
"""
Processing labeled Hyperion data.

Constructing batches from labeled data for subsequent network training and 
predicting

"""
# Authors: Jingge Xiao <jingge.xiao@gmail.com>
#
# Created on Fri Jul 28 10:21:50 2017

import scipy.io as sio
import scipy.interpolate as sip
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# original file path
mat_str = r'E:\Research\HyperspectralImageClassification\Experiment\Data\Labeled\IndianPines\Indian_pines_corrected.mat'

# paths of files containing spectral information for spectral resampling
ip_spec_file = r'E:\Research\HyperspectralImageClassification\Experiment\Data\Unlabeled\ing\IndianPines.csv'
hyper_spec_file = r'E:\Research\HyperspectralImageClassification\Experiment\Data\Unlabeled\ing\Hyperion.csv'

# set path of output data
npy_str = r'E:\Research\HyperspectralImageClassification\Experiment\Data\npy\ip'


# set batch size, which have to be same as that of unlabeled data.
img_size = 15


# read original data
mat_contents = sio.loadmat(mat_str)
original_image = mat_contents['indian_pines_corrected']

# get spectral information, positions that have spectral values
ip_point_list = []
ip_csv_reader = csv.reader(open(ip_spec_file, encoding='utf-8'))
for row in ip_csv_reader:
    ip_point_list.append(float(row[0]))
ip_point_array=np.array(ip_point_list)

hyper_point_list = []
hyper_csv_reader = csv.reader(open(hyper_spec_file, encoding='utf-8'))
for row in hyper_csv_reader:
    hyper_point_list.append(float(row[0]))
hyper_point_array=np.array(hyper_point_list)

# spectral resampling
global_image = np.zeros((original_image.shape[0],original_image.shape[1],len(hyper_point_array)), dtype = np.int)
for r in range(0,original_image.shape[0]):
    for c in range(0, original_image.shape[1]):
        ip_spec_array = original_image[r,c,:]
        f = sip.interp1d(ip_point_array, ip_spec_array)
        global_image[r,c,:] = f(hyper_point_array)

# fill the margin area with zero
mar_size = int((img_size-1)/2)
global_rows = global_image.shape[0]
global_cols = global_image.shape[1]
global_spes = global_image.shape[2]
larger_image = np.zeros((global_rows + mar_size*2, global_cols + mar_size*2, global_spes), dtype=np.int)
larger_image[mar_size: global_rows + mar_size, mar_size: global_cols + mar_size, :] = global_image
for p in range(0, mar_size):
    larger_image[p, mar_size: global_cols + mar_size, :] = larger_image[mar_size, mar_size: global_cols + mar_size, :]
    larger_image[global_rows + mar_size + p, mar_size: global_cols + mar_size, :] = larger_image[global_rows + mar_size - 1, mar_size: global_cols + mar_size, :]

for q in range(0, mar_size):
    larger_image[0 : global_rows + mar_size*2, q, :] = larger_image[0 : global_rows + mar_size*2, mar_size, :]
    larger_image[0 : global_rows + mar_size*2, global_cols + mar_size + q, :] = larger_image[0 : global_rows + mar_size*2, global_cols + mar_size - 1, :]
    

larger_rows = larger_image.shape[0]
larger_cols = larger_image.shape[1]

# construct pixel batchs
list_sub_image = []
npy_cnt = 0
for i in range(0, larger_rows):
    for j in range(0, larger_cols):
        sub_image = larger_image[i:i+img_size, j:j+img_size, : ]
        sub_image = sub_image.reshape(1, img_size, img_size, sub_image.shape[2])
        list_sub_image.append(sub_image)
        
        # limit the length of lists in order to avoid much too large npy files
        if len(list_sub_image) == 1000:
            npy_cnt = npy_cnt + 1
            array_sub_image=np.concatenate(list_sub_image, axis=0)
            
            # output array to file
            np.save(os.path.join(npy_str, str(npy_cnt)+'.npy'), array_sub_image)
            list_sub_image = []
            print(npy_cnt)
        if j == larger_cols - img_size:
            break
        
    if i == larger_cols - img_size:
        break
    
# output the last list to file 
array_sub_image=np.concatenate(list_sub_image, axis=0)
np.save(os.path.join(npy_str, 'last.npy'), array_sub_image)


# draw the picture of the data
# This is not needed for data preprocess, if you want to have a look of the 
# labeled data, you can modify and use these lines.
mx_data = larger_image[0: larger_rows, 0: larger_cols, 0 ]
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(223)
cmap=plt.cm.hot
im=ax.imshow(mx_data,cmap=cmap)  
plt.colorbar(im)

