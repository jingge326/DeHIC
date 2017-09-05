# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 00:07:40 2017

@author: alienware
"""

import numpy as np
import os
import random

path_ip_labels = r"E:\DeepLearning\Exp\data\original\Labeled\IndianPines\lable.npy"

path_ip_origin = r"E:\DeepLearning\Exp\data\original\Labeled\IndianPines"

array_ip_lab = np.load(path_ip_labels)

array_num_class = np.zeros((16, 2), dtype = int)
array_num_class[0:16,0] = np.arange(1, 17, 1)

bag_pos = []
for i_value in np.arange(0, 16):
    list_pos = []
    for i_pos in range(0, len(array_ip_lab)):
        lab_value = array_ip_lab[i_pos]
        if lab_value == i_value + 1:
            list_pos.append(i_pos)
            
    i_num = len(list_pos)
    array_num_class[i_value,1] = i_num

    # indices of bag plus one equals label values    
    bag_pos.append(list_pos)

np.save(os.path.join(path_ip_origin, 'pos_classes.npy'), bag_pos)
np.save(os.path.join(path_ip_origin, 'array_num_class.npy'), array_num_class)

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
np.save(os.path.join(path_ip_origin, 'train_samples_pos.npy'), list_train_pos)
np.save(os.path.join(path_ip_origin, 'validate_samples_pos.npy'), list_validate_pos)
