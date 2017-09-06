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

path_ip_3d = r"M:\DeepLearning\Exp\data\ing\3d\ip"

path_unlab_patches = r"M:\DeepLearning\Exp\data\ing\3d\hyper\unlabel_batches.npy"
path_lab_patches = os.path.join(path_ip_3d, "patches_ip.npy")

path_cae_save = os.path.join(path_ip_3d, "cae_model_3d_11.h5")
path_encoder_save = os.path.join(path_ip_3d, "encoder_model_3d_11.h5")
path_cae_lab_save = os.path.join(path_ip_3d, "cae_model_lab_3d_11.h5")
path_encoder_lab_save = os.path.join(path_ip_3d, "encoder_model_lab_3d_11.h5")

path_x_train = os.path.join(path_ip_3d, "training_3d_conv_x.npy")
path_y_train = os.path.join(path_ip_3d, "training_3d_conv_y.npy")
path_x_test = os.path.join(path_ip_3d, "validation_3d_conv_x.npy")
path_y_test = os.path.join(path_ip_3d, "validation_3d_conv_y.npy")

input_img = Input(shape=(1, 180, 8, 8))

x = Conv3D(4, (7, 3, 3), activation='relu', padding='same', data_format = "channels_first")(input_img)
x = MaxPooling3D((5, 2, 2), padding='same', data_format = "channels_first")(x)
x = Conv3D(6, (5, 3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = MaxPooling3D((3, 2, 2), padding='same', data_format = "channels_first")(x)
x = Conv3D(8, (3, 3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = MaxPooling3D((3, 2, 2), padding='same', data_format = "channels_first")(x)
x = Conv3D(10, (3, 1, 1), activation='relu', padding='same', data_format = "channels_first")(x)
x = MaxPooling3D((2, 1, 1), padding='same', data_format = "channels_first")(x)
x = Conv3D(12, (3, 1, 1), activation='relu', padding='same', data_format = "channels_first")(x)
encoded = MaxPooling3D((2, 1, 1), padding='same', data_format = "channels_first")(x)

# at this point the representation is (11, 1, 1, 1) i.e. 11-dimensional
x = UpSampling3D((2, 1, 1), data_format = "channels_first")(encoded)
x = Conv3D(12, (3, 1, 1), activation='relu', padding='same', data_format = "channels_first")(x)
x = UpSampling3D((2, 1, 1), data_format = "channels_first")(x)
x = Conv3D(10, (3, 1, 1), activation='relu', padding='same', data_format = "channels_first")(x)
x = UpSampling3D((3, 2, 2), data_format = "channels_first")(x)
x = Conv3D(8, (3, 3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = UpSampling3D((3, 2, 2), data_format = "channels_first")(x)
x = Conv3D(6, (5, 3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = UpSampling3D((5, 2, 2), data_format = "channels_first")(x)
x = Conv3D(4, (7, 3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
decoded = Conv3D(1, (7, 3, 3), activation='tanh', padding='same', data_format = "channels_first")(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='sgd', loss='mean_squared_error', metrics = ['accuracy'])

# Setting when to stop training
early_stopping = EarlyStopping(monitor='loss', patience=5)

# Training with unlabeled data
array_patches = np.load(path_unlab_patches)

t1 = time.time()

cae = autoencoder.fit(x=array_patches[0:130000,:,:,:,:], y=array_patches[0:130000,:,:,:,:], batch_size=100, epochs=250, callbacks=[early_stopping])

t2 = time.time()
t2_1 = t2 - t1


# Save the trained model
autoencoder.save(path_cae_save)
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
encoder.save(path_encoder_save)


# Train again using labeled data
array_batches_lab = np.load(path_lab_patches)
t3 = time.time()
cae = autoencoder.fit(x=array_batches_lab, y=array_batches_lab, batch_size=100, 
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