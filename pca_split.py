# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 12:40:07 2017

@author: XJG
"""
import sklearn.preprocessing as sp
from sklearn.decomposition import PCA
import scipy.io as sio
import numpy as np
import os

path_mat = r'M:\DeepLearning\Exp\data\Labeled\IndianPines\Indian_pines_corrected.mat'

hyper_lab_str = r"M:\DeepLearning\Exp\data\npy\ip\original_data\lable.npy"

sub_samples_str = r"M:\DeepLearning\Exp\data\npy\ip"

# Read original data
mat_contents = sio.loadmat(path_mat)
original_image = mat_contents['indian_pines_corrected']

# Transpose the array to "channle first"
original_image = original_image.transpose((2, 0, 1))

# Expand the array for scale
array_expand = original_image[:,0,:]
for i_row in range(1, original_image.shape[1]):
    tempmatirx = original_image[:,i_row,:]
    array_expand = np.hstack((array_expand,tempmatirx))
        
# Data normalization
array_expand_scaled = sp.scale(array_expand.T)

pca = PCA(n_components = 3)
array_pca = pca.fit_transform(array_expand_scaled)


# Split Conv dataset
x_train = []
y_train = []
x_test = []
y_test = []
list_train_pos = np.load(os.path.join(sub_samples_str, 'train_samples_pos.npy'))
list_validate_pos = np.load(os.path.join(sub_samples_str, 'validate_samples_pos.npy'))
array_hyper_lab = np.load(hyper_lab_str)

for i_cla_lab in np.arange(0, len(list_train_pos)):
    bags_pos_train = list_train_pos[i_cla_lab]
    bags_pos_validate = list_validate_pos[i_cla_lab]
    for i_pos_tra in bags_pos_train:
        x_train.append(array_pca[i_pos_tra,:])
        y_train.append(array_hyper_lab[i_pos_tra])
    
    for i_pos_val in bags_pos_validate:
        x_test.append(array_pca[i_pos_val,:])
        y_test.append(array_hyper_lab[i_pos_val])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

np.save(os.path.join(sub_samples_str, 'training_pca_x.npy'), x_train)
np.save(os.path.join(sub_samples_str, 'training_pca_y.npy'), y_train)
np.save(os.path.join(sub_samples_str, 'validation_pca_x.npy'), x_test)
np.save(os.path.join(sub_samples_str, 'validation_pca_y.npy'), y_test)
