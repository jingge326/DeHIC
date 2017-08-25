'''
Normally, in the decoder, the final
projection to the space where the image is reconstructed is linear, however
this does not have to be the case for a residual block as the degree to which
its output is linear or non-linear is determined by the data it is fed.
However, in order to cap the reconstruction in this example, a hard softmax is
applied as a bias because we know the MNIST digits are mapped to [0,1].
'''
from __future__ import print_function
import numpy as np

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Activation
from keras.layers import UpSampling2D, Conv2D, MaxPooling2D
from keras.layers import Input, BatchNormalization, ELU
import matplotlib.pyplot as plt
import keras.backend as K
from keras import layers


def convresblock(x, nfeats=8, ksize=3, nskipped=2, elu=True):
    y0 = Conv2D(nfeats, ksize, padding='same')(x)
    y = y0
    for i in range(nskipped):
        if elu:
            y = ELU()(y)
        else:
            y = BatchNormalization(axis=1)(y)
            y = Activation('relu')(y)
        y = Conv2D(nfeats, 1, padding='same')(y)
    return layers.add([y0, y])


# This example assume 'channels_first' data format.
K.set_image_data_format('channels_first')

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# The size of the kernel used for the MaxPooling2D
pool_size = 2
# The total number of feature maps at each layer
nfeats = [8, 16, 32, 64, 128]
# The sizes of the pooling kernel at each layer
pool_sizes = np.array([1, 1, 1, 1, 1]) * pool_size
# The convolution kernel size
ksize = 3
# Number of epochs to train for
epochs = 5
# Batch size during training
batch_size = 128

if pool_size == 2:
    # if using a 5 layer net of pool_size = 2
    x_train = np.pad(x_train, [[0, 0], [0, 0], [2, 2], [2, 2]],
                     mode='constant')
    x_test = np.pad(x_test, [[0, 0], [0, 0], [2, 2], [2, 2]], mode='constant')
    nlayers = 5
elif pool_size == 3:
    # if using a 3 layer net of pool_size = 3
    x_train = x_train[:, :, :-1, :-1]
    x_test = x_test[:, :, :-1, :-1]
    nlayers = 3
else:
    import sys
    sys.exit('Script supports pool_size of 2 and 3.')

# Shape of input to train on (note that model is fully convolutional however)
input_shape = x_train.shape[1:]
# The final list of the size of axis=1 for all layers, including input
nfeats_all = [input_shape[0]] + nfeats

# First build the encoder, all the while keeping track of the 'where' masks
img_input = Input(shape=input_shape)

# We push the 'where' masks to the following list
wheres = [None] * nlayers
y = img_input
for i in range(nlayers):
    y_prepool = convresblock(y, nfeats=nfeats_all[i + 1], ksize=ksize)
    y = MaxPooling2D(pool_size=(pool_sizes[i], pool_sizes[i]))(y_prepool)

# Now build the decoder, and use the stored 'where' masks to place the features
for i in range(nlayers):
    ind = nlayers - 1 - i
    y = UpSampling2D(size=(pool_sizes[ind], pool_sizes[ind]))(y)
    y = convresblock(y, nfeats=nfeats_all[ind], ksize=ksize)

# Use hard_simgoid to clip range of reconstruction
y = Activation('hard_sigmoid')(y)

# Define the model and it's mean square error loss, and compile it with Adam
model = Model(img_input, y)
model.compile('adam', 'mse')

# Fit the model
model.fit(x_train, x_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, x_test))

# Plot
x_recon = model.predict(x_test[:25])
x_plot = np.concatenate((x_test[:25], x_recon), axis=1)
x_plot = x_plot.reshape((5, 10, input_shape[-2], input_shape[-1]))
x_plot = np.vstack([np.hstack(x) for x in x_plot])
plt.figure()
plt.axis('off')
plt.title('Test Samples: Originals/Reconstructions')
plt.imshow(x_plot, interpolation='none', cmap='gray')
plt.savefig('reconstructions.png')
