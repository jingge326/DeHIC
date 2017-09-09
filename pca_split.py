#-*-coding:UTF-8-*-
"""
Basic process of Indian Pines data set for PCA-based classification

Including:
Normalizing pixel values within each band
PCA transformation
Splitting data into training and validation sets according to selected positions

"""
# Authors: Jingge Xiao <jingge.xiao@gmail.com>
#
# Created on Fri Jul 28 10:21:50 2017

import sklearn.preprocessing as sp
from sklearn.decomposition import PCA
import scipy.io as sio
import numpy as np
import os

num_com = 12

path_ip = r"E:\DeepLearning\Exp\data\original\Labeled\IndianPines"

path_pca = r"E:\DeepLearning\Exp\data\ing\pca"

path_mat = os.path.join(path_ip, "Indian_pines_corrected.mat")

path_ip_labels = os.path.join(path_ip, "lable.npy")

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

pca = PCA(n_components = num_com)
array_pca = pca.fit_transform(array_expand_scaled)


# Split Conv dataset
x_train = []
y_train = []
x_test = []
y_test = []
list_train_pos = np.load(os.path.join(path_ip, 'train_samples_pos.npy'))
list_validate_pos = np.load(os.path.join(path_ip, 'validate_samples_pos.npy'))
array_ip_lab = np.load(path_ip_labels)

for i_cla_lab in np.arange(0, len(list_train_pos)):
    bags_pos_train = list_train_pos[i_cla_lab]
    bags_pos_validate = list_validate_pos[i_cla_lab]
    for i_pos_tra in bags_pos_train:
        x_train.append(array_pca[i_pos_tra,:])
        y_train.append(array_ip_lab[i_pos_tra])
    
    for i_pos_val in bags_pos_validate:
        x_test.append(array_pca[i_pos_val,:])
        y_test.append(array_ip_lab[i_pos_val])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

np.save(os.path.join(path_pca, 'training_pca_x_' + str(num_com)), x_train)
np.save(os.path.join(path_pca, 'training_pca_y_' + str(num_com)), y_train)
np.save(os.path.join(path_pca, 'validation_pca_x_' + str(num_com)), x_test)
np.save(os.path.join(path_pca, 'validation_pca_y_' + str(num_com)), y_test)
