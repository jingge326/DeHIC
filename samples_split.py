# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 00:07:40 2017

@author: alienware
"""

import numpy as np
import os

ip_scaled_str = r'D:\DeepLearning\Exp\data\Labeled\IndianPines\Indian_pines_corrected_scaled.npy'
ip_lab_str = r"C:\DeepLearning\Exp\data\npy\ip\lable.npy"

array_ip_scaled = np.load(ip_scaled_str)
array_ip_lab = np.load(ip_lab_str)

x_train_bands = []
y_train_bands = []
x_test_bands = []
y_test_bands = []

arr_pix_bands = []
for i_row in array_ip_scaled.shape[1]:
   for i_col in array_ip_scaled.shape[2]:
     arr_pix_bands.append(array_ip_scaled[:, i_row, i_col])  

arr_pix_bands = np.array(arr_pix_bands)
    
for i_cla_lab in np.arange(0, len(list_train_pos)):
    bags_pos_train = list_train_pos[i_cla_lab]
    bags_pos_validate = list_validate_pos[i_cla_lab]
    for i_pos_tra in bags_pos_train:
        x_train.append(array_batches_hyper[i_pos_tra,:,:,:])
        y_train.append(array_hyper_lab[i_pos_tra])
    
    for i_pos_val in bags_pos_validate:
        x_test.append(array_batches_hyper[i_pos_val,:,:,:])
        y_test.append(array_hyper_lab[i_pos_val])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

np.save(os.path.join(sub_samples_str, 'training_conv_x.npy'), x_train)
np.save(os.path.join(sub_samples_str, 'training_conv_y.npy'), y_train)
np.save(os.path.join(sub_samples_str, 'validation_conv_x.npy'), x_test)
np.save(os.path.join(sub_samples_str, 'validation_conv_y.npy'), y_test)
