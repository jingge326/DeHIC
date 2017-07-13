import numpy as np
import gdal
import os
import gc
import matplotlib.pyplot as plt

#set batch size, must be an odd number
img_size = 15

npy_str = r'G:\npy'
tif_str = r'E:\Research\HyperspectralImageClassification\Experiment\Data\Unlabeled\ing\tif'
fileObjectsList = []
for i in os.walk(tif_str):
    for j in i[2]:
        fileFormatStr = j[-3:]
        if fileFormatStr == 'tif':
            fileObjectsList.append(j)

list_sub_image = []
num_list = 1
for f in fileObjectsList:
    print(f)
    imagery_path = os.path.join(tif_str, f)

    #read image
    dataset = gdal.Open(imagery_path)
    dsmatrix = dataset.ReadAsArray(xoff=0, yoff=0, xsize=dataset.RasterXSize, ysize=dataset.RasterYSize)
    image_rows = dsmatrix.shape[1]
    image_cols = dsmatrix.shape[2]    
    image_spec = dsmatrix.shape[0]
    
    #construct pixel batchs
    for i in range(0, image_rows, img_size):
        if i >= image_rows - img_size:
            break
        for j in range(0, image_cols, img_size):
            if j >= image_cols - img_size:
                break
            break_flag = False
            sub_image = dsmatrix[:, i:i+img_size, j:j+img_size]
            my_sub_image = np.zeros((img_size, img_size, image_spec), dtype = np.int)
            for p in range(0, my_sub_image.shape[0]):
                for q in range(0, my_sub_image.shape[1]):
                    my_sub_image[p, q, :] = sub_image[:, p, q]
                    if np.mean(my_sub_image[p, q, :]) == 0:
                        break_flag = True
                        break
                if break_flag == True:
                    break
            if break_flag == True:
                continue
            my_sub_image = my_sub_image.reshape(1, img_size, img_size, my_sub_image.shape[2])
            list_sub_image.append(my_sub_image)
            if(len(list_sub_image)==1000):
                array_sub_image=np.concatenate(list_sub_image, axis=0)
                #output array to file
                np.save(os.path.join(npy_str, str(num_list)+'.npy'), array_sub_image)
                num_list = num_list + 1
                list_sub_image = []

    #collect memory
    del(dsmatrix)
    gc.collect()
    
array_sub_image=np.concatenate(list_sub_image, axis=0)
#output array to file
np.save(os.path.join(npy_str, 'last.npy'), array_sub_image)
 
#Schematic diagram of spectral curve
d_y = array_sub_image[300,0,0,:]
d_x = np.arange(0,len(array_sub_image[0,0,0,:]),1)
plt.plot(d_x, d_y)