from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense
from keras.models import Model
import keras
import numpy as np
import os
import time

num_epochs = 10

num_classes = 17

unlab_dir_str = r"C:\DeepLearning\Exp\data\npy\hyper"

str_cae_save = r'C:\DeepLearning\Exp\data\npy\ip\cae_model.h5'
str_encoder_save = r'C:\DeepLearning\Exp\data\npy\ip\encoder_model.h5'

path_x_train = r'C:\DeepLearning\Exp\data\npy\ip\training_conv_x.npy'
path_y_train = r'C:\DeepLearning\Exp\data\npy\ip\training_conv_y.npy'
path_x_test = r'C:\DeepLearning\Exp\data\npy\ip\validation_conv_x.npy'
path_y_test = r'C:\DeepLearning\Exp\data\npy\ip\validation_conv_y.npy'

input_img = Input(shape=(175, 8, 8))

x = Conv2D(88, (3, 3), activation='relu', padding='same', data_format = "channels_first")(input_img)
x = MaxPooling2D((2, 2), padding='same', data_format = "channels_first")(x)
x = Conv2D(44, (3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = MaxPooling2D((2, 2), padding='same', data_format = "channels_first")(x)
x = Conv2D(12, (2, 2), activation='relu', padding='same', data_format = "channels_first")(x)
encoded = MaxPooling2D((2, 2), padding='same', data_format = "channels_first")(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
x = UpSampling2D((2, 2), data_format = "channels_first")(encoded)
x = Conv2D(12, (2, 2), activation='relu', padding='same', data_format = "channels_first")(x)
x = UpSampling2D((2, 2), data_format = "channels_first")(x)
x = Conv2D(44, (3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = UpSampling2D((2, 2), data_format = "channels_first")(x)
x = Conv2D(88, (3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
decoded = Conv2D(175, (3, 3), activation='sigmoid', padding='same', data_format = "channels_first")(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='sgd', loss='mean_squared_error', metrics = ['accuracy'])

list_dir_strs = os.listdir(unlab_dir_str)
    
while num_epochs > 0:
    print("begin epoch: " + str(num_epochs))
    
    t1 = time.time()
    for name_str in list_dir_strs:
        npy_path = os.path.join(unlab_dir_str, name_str)
        array_batches = np.load(npy_path)
        
        cae = autoencoder.fit(array_batches, array_batches, batch_size=100)
        
    num_epochs = num_epochs - 1
    t2 = time.time()
    print('time' + str(t2-t1))

# Save the trained model
autoencoder.save(str_cae_save)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

encoder.save(str_encoder_save)


# fine turning

x_train = np.load(path_x_train)
y_train = np.load(path_y_train)
x_test = np.load(path_x_test)
y_test = np.load(path_y_test)

input_layer = autoencoder.get_layer(index = 0)
encoded_layer = autoencoder.get_layer(index = 6)

x = Flatten()(decoded)
out_class_layer = Dense(num_classes, activation = 'softmax')(x)

classifier_encoder = Model(input_img, out_class_layer)
classifier_encoder.compile(optimizer='sgd', loss='mean_squared_error', metrics = ['accuracy'])

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


classifier_encoder.fit(x_train, y_train, epochs = 10, validation_data = (x_test, y_test))

y_predict = classifier_encoder.predict(x_test)
