#-*-coding:UTF-8-*-
"""
An example of 3d-convolutional autoencoder
The autoencoder network is pretrained by nearly 1.9 million unlabelled hyperion patches.
As for the classification of Indian Pines dataset, autoencoder network is trained 
again by Indian Pines dataset patches without label. Then the encoder part is extracted 
and used to produce encoded features. Finally, features generated from convolutional 
encoder is fed to SVM classifier.

"""
# Authors: Jingge Xiao <jingge.xiao@gmail.com>
#
# Created on Fri Jul 28 10:21:50 2017

import numpy as np
import os
import time
from sklearn import svm
from sklearn.metrics import cohen_kappa_score

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D
from keras.models import Model
from keras.callbacks import EarlyStopping

num_classes = 16
num_bands = 180

path_ip = r"M:\DeepLearning\Exp\data\ing\180\ip\basic"
path_model_save = r"M:\DeepLearning\Exp\data\ing\180\ip\3d_conv"

path_unlab_patches = r"M:\DeepLearning\Exp\data\ing\180\hyper\3d_conv\unlabel_patches.npy"

path_lab_patches = os.path.join(path_ip, "patches_ip.npy")

path_x_train = os.path.join(path_ip, "training_conv_x.npy")
path_y_train = os.path.join(path_ip, "training_conv_y.npy")
path_x_test = os.path.join(path_ip, "validation_conv_x.npy")
path_y_test = os.path.join(path_ip, "validation_conv_y.npy")

path_cae_save = os.path.join(path_model_save, "cae_model_3d_12.h5")
path_encoder_save = os.path.join(path_model_save, "encoder_model_3d_12.h5")
path_cae_lab_save = os.path.join(path_model_save, "cae_model_lab_3d_12.h5")
path_encoder_lab_save = os.path.join(path_model_save, "encoder_model_lab_3d_12.h5")

input_img = Input(shape=(1, num_bands, 8, 8))

x = Conv3D(4, (5, 3, 3), activation='relu', padding='same', data_format = "channels_first")(input_img)
x = MaxPooling3D((5, 2, 2), padding='same', data_format = "channels_first")(x)
x = Conv3D(8, (5, 3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = MaxPooling3D((4, 2, 2), padding='same', data_format = "channels_first")(x)
x = Conv3D(10, (3, 3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = MaxPooling3D((3, 2, 2), padding='same', data_format = "channels_first")(x)
x = Conv3D(12, (3, 1, 1), activation='relu', padding='same', data_format = "channels_first")(x)
encoded = MaxPooling3D((3, 1, 1), padding='same', data_format = "channels_first")(x)

# at this point the representation is (12, 1, 1, 1) i.e. 11-dimensional
x = UpSampling3D((3, 1, 1), data_format = "channels_first")(encoded)
x = Conv3D(12, (3, 1, 1), activation='relu', padding='same', data_format = "channels_first")(x)
x = UpSampling3D((3, 2, 2), data_format = "channels_first")(x)
x = Conv3D(10, (3, 3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = UpSampling3D((4, 2, 2), data_format = "channels_first")(x)
x = Conv3D(8, (5, 3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = UpSampling3D((5, 2, 2), data_format = "channels_first")(x)
x = Conv3D(4, (5, 3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
decoded = Conv3D(1, (1, 1, 1), activation='tanh', padding='same', data_format = "channels_first")(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='sgd', loss='mean_squared_error', metrics = ['accuracy'])

# Setting when to stop training
early_stopping = EarlyStopping(monitor='loss', patience=5)

# Training with unlabelled data
array_patches = np.load(path_unlab_patches)

t1 = time.time()

cae = autoencoder.fit(x=array_patches, y=array_patches, batch_size=100, 
                      epochs=250, callbacks=[early_stopping])

t2 = time.time()
t2_1 = t2 - t1


# Save the trained model
autoencoder.save(path_cae_save)
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
encoder.save(path_encoder_save)


# Train again using labeled data
array_patches_lab = np.load(path_lab_patches)
t3 = time.time()
cae = autoencoder.fit(x=array_patches_lab, y=array_patches_lab, batch_size=100, 
                      epochs=250, callbacks=[early_stopping])
t4 = time.time()
t4_3 = t4 - t3


# Save the trained model
autoencoder.save(path_cae_lab_save)
# This model maps an input to its encoded representation
encoder = Model(input_img, encoded)
encoder.save(path_encoder_lab_save)


# Loading labeled samples
x_train = np.load(path_x_train)
y_train = np.load(path_y_train)
x_test = np.load(path_x_test)
y_test = np.load(path_y_test)

# Produce encoded features
x_train_co = encoder.predict(x_train)
x_test_co = encoder.predict(x_test)          
x_train_co = x_train_co.reshape((x_train_co.shape[0], x_train_co.shape[1]))
x_test_co = x_test_co.reshape((x_test_co.shape[0], x_test_co.shape[1]))

# Train the classifier
clf_svm = svm.SVC()
clf_svm.fit(x_train_co, y_train)

# Predict lables based on image data
y_predict=clf_svm.predict(x_test_co)
kappa_value = cohen_kappa_score(y_predict, y_test)

print(kappa_value)
print(t2_1)
print(t4_3)