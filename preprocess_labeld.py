#-*-coding:UTF-8-*-
"""
Processing labeled Hyperion data.

Constructing batches from labeled data for subsequent network training and 
predicting

"""
# Authors: Jingge Xiao <jingge.xiao@gmail.com>
#
# Created on Fri Jul 28 10:21:50 2017

import scipy.interpolate as sip
import numpy as np
import csv
import os

# original file path
str_npy_in = r'D:\DeepLearning\Exp\data\Labeled\IndianPines\Indian_pines_corrected_scaled.npy'

# paths of files containing spectral information for spectral resampling
ip_spec_file = r'D:\DeepLearning\Exp\data\Unlabled\IndianPines.csv'
hyper_spec_file = r'D:\DeepLearning\Exp\data\Unlabled\Hyperion.csv'

# set path of output data
str_npy_path = r'C:\DeepLearning\Exp\data\npy\ip'


# set batch size, which have to be same as that of unlabeled data.
img_size = 8

# read original data
original_image = np.load(str_npy_in)
original_image = original_image.transpose((2, 0, 1))

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
global_image = np.zeros((len(hyper_point_array), original_image.shape[1],original_image.shape[2]))
for r in range(0,original_image.shape[1]):
    for c in range(0, original_image.shape[2]):
        ip_spec_array = original_image[:,r,c]
        f = sip.interp1d(ip_point_array, ip_spec_array)
        global_image[:,r,c] = f(hyper_point_array)

# change this equation if change img_size
mar_size = int(img_size/2)

# fill the marginal area with values in borders
global_rows = global_image.shape[1]
global_cols = global_image.shape[2]
global_spes = global_image.shape[0]
larger_image = np.zeros((global_spes, global_rows + mar_size*2, global_cols + mar_size*2))
larger_image[:, mar_size: global_rows + mar_size, mar_size: global_cols + mar_size] = global_image
for p in range(0, mar_size):
    larger_image[:, p, mar_size: global_cols + mar_size] = larger_image[:, mar_size, mar_size: global_cols + mar_size]
    larger_image[:, global_rows + mar_size + p, mar_size: global_cols + mar_size] = larger_image[:, global_rows + mar_size - 1, mar_size: global_cols + mar_size]

for q in range(0, mar_size):
    larger_image[:, 0 : global_rows + mar_size*2, q] = larger_image[:, 0 : global_rows + mar_size*2, mar_size]
    larger_image[:, 0 : global_rows + mar_size*2, global_cols + mar_size + q] = larger_image[:, 0 : global_rows + mar_size*2, global_cols + mar_size - 1]
    

larger_rows = larger_image.shape[1]
larger_cols = larger_image.shape[2]

# construct pixel batchs
list_sub_image = []
npy_cnt = 0
for i in range(0, larger_rows):
    for j in range(0, larger_cols):
        if i < larger_cols - img_size:
            if j < larger_cols - img_size:        
                sub_image = larger_image[:, i:i+img_size, j:j+img_size]              
                t_vector = sub_image[:, int(img_size/2) -1, int(img_size/2) - 1]
                t_cube = np.array([[t_vector, t_vector], [t_vector, t_vector]]).transpose((2, 0, 1))
                sub_image[:, int(img_size/2) -1: int(img_size/2) +1, int(img_size/2) - 1:int(img_size/2) +1] = t_cube
                
                list_sub_image.append(sub_image)
                
        #        # limit the length of lists in order to avoid much too large npy files
        #        if len(list_sub_image) == 10000:
        #            npy_cnt = npy_cnt + 1
        #            array_sub_image=np.array(list_sub_image)
        #            
        #            # output array to file
        #            np.save(os.path.join(str_npy_path, str(npy_cnt)+'.npy'), array_sub_image)
        #            list_sub_image = []
        #            print(npy_cnt)
            
    
# output the last list to file 
array_sub_image=np.array(list_sub_image)
np.save(os.path.join(str_npy_path, 'last.npy'), array_sub_image)
