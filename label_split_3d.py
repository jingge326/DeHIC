# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 00:07:40 2017

@author: alienware
"""

import numpy as np
import os

# Set input path of 3d conv patches
path_ip_batches = r"E:\DeepLearning\Exp\data\ing\3d\ip\patches_ip.npy"

# Set input path of common files
path_ip_labels = r"E:\DeepLearning\Exp\data\original\Labeled\IndianPines\lable.npy"
path_train_pos = r"E:\DeepLearning\Exp\data\original\Labeled\IndianPines\train_samples_pos.npy"
path_validate_pos = r"E:\DeepLearning\Exp\data\original\Labeled\IndianPines\validate_samples_pos.npy"

# Set output path
path_ip_3d = r"E:\DeepLearning\Exp\data\ing\3d\ip"

array_ip_batches = np.load(path_ip_batches)
array_ip_lab = np.load(path_ip_labels)
array_train_pos = np.load(path_train_pos)
array_validate_pos = np.load(path_validate_pos)

# Split Conv dataset
x_train = []
y_train = []
x_test = []
y_test = []
for i_cla_lab in np.arange(0, array_train_pos.shape[0]):
    list_pos_train = array_train_pos[i_cla_lab]
    list_pos_validate = array_validate_pos[i_cla_lab]
    for i_pos_tra in list_pos_train:
        x_train.append(array_ip_batches[i_pos_tra,:,:,:])
        y_train.append(array_ip_lab[i_pos_tra])
    
    for i_pos_val in list_pos_validate:
        x_test.append(array_ip_batches[i_pos_val,:,:,:])
        y_test.append(array_ip_lab[i_pos_val])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

np.save(os.path.join(path_ip_3d, 'training_3d_conv_x.npy'), x_train)
np.save(os.path.join(path_ip_3d, 'training_3d_conv_y.npy'), y_train)
np.save(os.path.join(path_ip_3d, 'validation_3d_conv_x.npy'), x_test)
np.save(os.path.join(path_ip_3d, 'validation_3d_conv_y.npy'), y_test)
