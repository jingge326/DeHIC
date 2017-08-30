# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 00:07:40 2017

@author: alienware
"""

import numpy as np
import os
import random

hyper_data_str = r"G:\DeepLearning\Exp\data\npy\ip\original_data\batches_ip.npy"

hyper_lab_str = r"G:\DeepLearning\Exp\data\npy\ip\original_data\lable.npy"

sub_samples_str = r"G:\DeepLearning\Exp\data\npy\ip"

array_batches_hyper = np.load(hyper_data_str)
array_hyper_lab = np.load(hyper_lab_str)

array_num_class = np.zeros((16, 2), dtype = int)
array_num_class[0:16,0] = np.arange(1, 17, 1)

bag_pos = []
bag_sub_samples = []
for i_value in np.arange(0, 16):
    list_sub_samples = []
    list_pos = []
    for i_pos in range(0, len(array_hyper_lab)):
        lab_value = array_hyper_lab[i_pos]
        if lab_value == i_value + 1:
            list_sub_samples.append(array_batches_hyper[i_pos,:,:,:])
            list_pos.append(i_pos)
            
    i_num = len(list_sub_samples)
    array_num_class[i_value,1] = i_num

    # indices of bag plus one equals label values    
    bag_pos.append(list_pos)
    bag_sub_samples.append(list_sub_samples)

np.save(os.path.join(sub_samples_str, 'pos_classes.npy'), bag_pos)
np.save(os.path.join(sub_samples_str, 'sub_samples.npy'), bag_sub_samples)
np.save(os.path.join(sub_samples_str, 'array_num_class.npy'), array_num_class)

# Produce wanted numbers list
list_wanted_num = []

## Way No.1 by rate
#train_rate = 0.2
#for i_num in array_num_class[0:17,1]:
#    wanted_num = int(i_num*train_rate)
#    list_wanted_num.append(wanted_num)

# Way No.2 by given number
wanted_num = 50
sec_wanted_num = 15
for i_num in array_num_class[0:16,1]:
    if i_num > wanted_num:
        list_wanted_num.append(wanted_num)
    else:
        list_wanted_num.append(sec_wanted_num)

list_train_pos = []
list_validate_pos = []
for i_num_pos in np.arange(0, len(list_wanted_num)):
    train_pos = random.sample(bag_pos[i_num_pos], list_wanted_num[i_num_pos])
    list_train_pos.append(train_pos)
    validate_pos = list(set(bag_pos[i_num_pos])^set(train_pos))
    list_validate_pos.append(validate_pos)
    
# Output list of positions
np.save(os.path.join(sub_samples_str, 'train_samples_pos.npy'), list_train_pos)
np.save(os.path.join(sub_samples_str, 'validate_samples_pos.npy'), list_validate_pos)

# Split Conv dataset
x_train = []
y_train = []
x_test = []
y_test = []
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
