import numpy as np
import os
import time
from sklearn import svm
from sklearn.metrics import cohen_kappa_score
from keras.callbacks import EarlyStopping

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

num_classes = 16

unlab_dir_str = r"M:\DeepLearning\Exp\data\npy\hyper\list"

path_unlab_batches = r"M:\DeepLearning\Exp\data\npy\hyper\unlabel_batches.npy"
path_lab_batches = r"M:\DeepLearning\Exp\data\npy\ip\original_data\batches_ip.npy"

path_cae_save = r'M:\DeepLearning\Exp\data\npy\ip\cae_model.h5'
path_encoder_save = r'M:\DeepLearning\Exp\data\npy\ip\encoder_model.h5'
path_cae_lab_save = r'M:\DeepLearning\Exp\data\npy\ip\cae_model_lab.h5'
path_encoder_lab_save = r'M:\DeepLearning\Exp\data\npy\ip\encoder_model_lab.h5'

path_x_train = r'M:\DeepLearning\Exp\data\npy\ip\training_conv_x.npy'
path_y_train = r'M:\DeepLearning\Exp\data\npy\ip\training_conv_y.npy'
path_x_test = r'M:\DeepLearning\Exp\data\npy\ip\validation_conv_x.npy'
path_y_test = r'M:\DeepLearning\Exp\data\npy\ip\validation_conv_y.npy'

input_img = Input(shape=(175, 8, 8))

x = Conv2D(122, (3, 3), activation='relu', padding='same', data_format = "channels_first")(input_img)
x = Conv2D(88, (3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = MaxPooling2D((2, 2), padding='same', data_format = "channels_first")(x)
x = Conv2D(44, (3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = Conv2D(22, (3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = MaxPooling2D((2, 2), padding='same', data_format = "channels_first")(x)
x = Conv2D(11, (2, 2), activation='relu', padding='same', data_format = "channels_first")(x)
x = Conv2D(6, (2, 2), activation='relu', padding='same', data_format = "channels_first")(x)
encoded = MaxPooling2D((2, 2), padding='same', data_format = "channels_first")(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
x = UpSampling2D((2, 2), data_format = "channels_first")(encoded)
x = Conv2D(6, (2, 2), activation='relu', padding='same', data_format = "channels_first")(x)
x = Conv2D(11, (2, 2), activation='relu', padding='same', data_format = "channels_first")(x)
x = UpSampling2D((2, 2), data_format = "channels_first")(x)
x = Conv2D(22, (2, 2), activation='relu', padding='same', data_format = "channels_first")(x)
x = Conv2D(44, (3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = UpSampling2D((2, 2), data_format = "channels_first")(x)
x = Conv2D(88, (3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = Conv2D(122, (3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
decoded = Conv2D(175, (3, 3), activation='sigmoid', padding='same', data_format = "channels_first")(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='sgd', loss='mean_squared_error', metrics = ['accuracy'])

# Setting when to stop training
early_stopping = EarlyStopping(monitor='loss', patience=2)

# Training with unlabeled data
array_batches = np.load(path_unlab_batches)
t1 = time.time()

#cae = autoencoder.fit(x=array_batches, y=array_batches, batch_size=100, 
#                      epochs=200, callbacks=[early_stopping])

cae = autoencoder.fit(x=array_batches, y=array_batches, batch_size=100, epochs=250)

t2 = time.time()
t2_1 = t2 - t1


# Save the trained model
autoencoder.save(path_cae_save)
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
encoder.save(path_encoder_save)


# Train again using labeled data
array_batches_lab = np.load(path_lab_batches)
t3 = time.time()
cae = autoencoder.fit(x=array_batches_lab, y=array_batches_lab, batch_size=100, 
                      epochs=200, callbacks=[early_stopping])
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