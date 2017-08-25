from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D
from keras.models import Model
import numpy as np
import os

num_epochs = 2

unlab_dir_str = r"C:\DeepLearning\Exp\data\npy\hyper"

input_img = Input(shape=(1, 175, 11, 11))

x = Conv3D(16, (9, 3, 3), activation='relu', padding='same', data_format = "channels_first")(input_img)
x = MaxPooling3D((4, 2, 2), padding='same', data_format = "channels_first")(x)
x = Conv3D(8, (7, 3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = MaxPooling3D((2, 2, 2), padding='same', data_format = "channels_first")(x)
x = Conv3D(8, (5, 3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
encoded = MaxPooling3D((2, 2, 2), padding='same', data_format = "channels_first")(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv3D(8, (5, 3, 3), activation='relu', padding='same', data_format = "channels_first")(encoded)
x = UpSampling3D((2, 2, 2), data_format = "channels_first")(x)
x = Conv3D(8, (7, 3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = UpSampling3D((2, 2, 2), data_format = "channels_first")(x)
x = Conv3D(16, (9, 3, 3), activation='relu', data_format = "channels_first")(x)
x = UpSampling3D((4, 2, 2), data_format = "channels_first")(x)
decoded = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same', data_format = "channels_first")(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')



list_dir_strs = os.listdir(unlab_dir_str)
    
while num_epochs > 0:
    for name_str in list_dir_strs:
        npy_path = os.path.join(unlab_dir_str, name_str)
        array_batches = np.load(npy_path)
        
        s_list_batches = []
        for i_batch in array_batches:
            s_list_batches.append(i_batch.reshape(1, array_batches.shape[1], 
                                                  array_batches.shape[2], array_batches.shape[3]))
        
        s_array_batches = np.array(s_list_batches)
        
        cae = autoencoder.fit(s_array_batches, s_array_batches,
                    epochs=1,
                    batch_size=100,
                    shuffle=True)




decoded_imgs = autoencoder.predict(x_test)