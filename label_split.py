# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 00:07:40 2017

@author: alienware
"""

import numpy as np
import os
import random

hyper_data_str = r"C:\DeepLearning\Exp\data\npy\ip\last.npy"

hyper_lab_str = r"C:\DeepLearning\Exp\data\npy\ip\lable.npy"

sub_samples_str = r"C:\DeepLearning\Exp\data\npy\ip"

array_batches_hyper = np.load(hyper_data_str)
array_hyper_lab = np.load(hyper_lab_str)

train_rate = 0.2

num_class = np.zeros((17, 3), dtype = int)
num_class[0:17,0] = np.arange(0, 17, 1)
list_train_pos = []
list_validate_pos = []
for i_value in num_class[0:17,0]:
    list_sub_samples = []
    list_pos = []
    for i_pos in range(0, len(array_hyper_lab)):
        lab_value = array_hyper_lab[i_pos]
        if lab_value == i_value:
            list_sub_samples.append(array_batches_hyper[i_pos,:,:,:])
            list_pos.append(i_pos)
            
    i_num = len(list_sub_samples)
    wanted_num = int(i_num*train_rate)
    num_class[i_value,1] = i_num
    num_class[i_value,2] = wanted_num
    
    train_pos = random.sample(list_pos, wanted_num)
    list_train_pos.append(train_pos)
    validate_pos = list(set(list_pos)^set(train_pos))
    list_validate_pos.append(validate_pos)
    
    np.save(os.path.join(sub_samples_str, 'pos_class_'+str(i_value)+'.npy'), list_pos)
    np.save(os.path.join(sub_samples_str, 'samples_class_'+str(i_value)+'.npy'), list_sub_samples)
np.save(os.path.join(sub_samples_str, 'num_class.npy'), num_class)
np.save(os.path.join(sub_samples_str, 'train_samples_pos.npy'), list_train_pos)
np.save(os.path.join(sub_samples_str, 'validate_samples_pos.npy'), list_validate_pos)

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
