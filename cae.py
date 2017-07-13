from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D
from keras.models import Model
from keras.datasets import mnist
import numpy as np


(x_train, _), (x_test, _) = mnist.load_data(r'E:\Research\HyperspectralImageClassification\Experiment\Data\others\mnist.npz')
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv3D(16, (3, 3, 7), activation='relu', padding='same')(input_img)
x = MaxPooling3D((2, 2, 4), padding='same')(x)
x = Conv3D(8, (3, 3, 7), activation='relu', padding='same')(x)
x = MaxPooling3D((2, 2, 4), padding='same')(x)
x = Conv3D(8, (3, 3, 7), activation='relu', padding='same')(x)
encoded = MaxPooling3D((2, 2, 4), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv3D(8, (3, 3, 7), activation='relu', padding='same')(encoded)
x = UpSampling3D((2, 2, 4))(x)
x = Conv3D(8, (3, 3, 7), activation='relu', padding='same')(x)
x = UpSampling3D((2, 2, 4))(x)
x = Conv3D(16, (3, 3, 7), activation='relu')(x)
x = UpSampling3D((2, 2, 4))(x)
decoded = Conv3D(1, (3, 3, 7), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

cae = autoencoder.fit(x_train, x_train,
                epochs=3,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

decoded_imgs = autoencoder.predict(x_test)