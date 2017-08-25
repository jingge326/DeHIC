#-*-coding:UTF-8-*-
"""
Standardization of datasets.

Transform the data to center it by removing the mean value of each feature, 
then scale it by dividing non-constant features by their standard deviation.

All image must be scaled before subsequent classification progress

"""
# Authors: Jingge Xiao <jingge.xiao@gmail.com>
#
# Created on Fri Jul 28 10:21:50 2017

import sklearn.preprocessing as sp
import scipy.io as sio
import numpy as np

# original file path
mat_str = r'D:\DeepLearning\Exp\data\Labeled\IndianPines\Indian_pines_corrected.mat'

# set path of output data
npy_str = r'D:\DeepLearning\Exp\data\Labeled\IndianPines\Indian_pines_corrected_scaled.npy'

# read original data
mat_contents = sio.loadmat(mat_str)
original_image = mat_contents['indian_pines_corrected']

rows = original_image.shape[0]
cols = original_image.shape[1]

data_array = original_image[0,:,:]
for irow in range(1, rows):
    tempmatirx = original_image[irow,:,:]
    data_array = np.vstack((data_array,tempmatirx))
        
# Data normalization
data_array_scaled = sp.scale(data_array)

# Convert the result Frame to Matrix
result_matrix = np.empty_like(original_image, dtype = "float64")
for irow_b in range(0,rows):
    result_matrix[irow_b,:,:] = data_array_scaled[irow_b*cols:(irow_b+1)*cols,:]

np.save(npy_str, result_matrix)


# original file path
mat_lab_str = r'D:\DeepLearning\Exp\data\Labeled\IndianPines\Indian_pines_gt.mat'
str_path_lab = r'C:\DeepLearning\Exp\data\npy\ip\lable.npy'
# read original data
mat_lab_contents = sio.loadmat(mat_lab_str)
original_lable = mat_lab_contents['indian_pines_gt']
array_lab = original_lable.flatten(order='C')
np.save(str_path_lab, array_lab)