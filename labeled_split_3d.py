#-*-coding:UTF-8-*-
"""
Splitting Indian Pines data set into training and validation sets according to selected positions

This file is for 3d-convolutional networks.

"""
# Authors: Jingge Xiao <jingge.xiao@gmail.com>
#
# Created on Fri Jul 28 10:21:50 2017

import numpy as np
import os


# Set path patches
path_ip_3d = r"M:\DeepLearning\Exp\data\ing\180\ip\3d_conv"

# Path of input patches
path_ip_patches = os.path.join(path_ip_3d, "patches_ip.npy")

# Set path selected data positions
path_ip_base =  r"M:\DeepLearning\Exp\data\original\Labeled\IndianPines"
path_ip_labels = os.path.join(path_ip_base, "lable.npy")
path_train_pos = os.path.join(path_ip_base, "train_samples_pos.npy")
path_validate_pos = os.path.join(path_ip_base, "validate_samples_pos.npy")



array_ip_patches = np.load(path_ip_patches)
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
        x_train.append(array_ip_patches[i_pos_tra,:,:,:])
        y_train.append(array_ip_lab[i_pos_tra])
    
    for i_pos_val in list_pos_validate:
        x_test.append(array_ip_patches[i_pos_val,:,:,:])
        y_test.append(array_ip_lab[i_pos_val])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

np.save(os.path.join(path_ip_3d, 'training_3d_conv_x.npy'), x_train)
np.save(os.path.join(path_ip_3d, 'training_3d_conv_y.npy'), y_train)
np.save(os.path.join(path_ip_3d, 'validation_3d_conv_x.npy'), x_test)
np.save(os.path.join(path_ip_3d, 'validation_3d_conv_y.npy'), y_test)
