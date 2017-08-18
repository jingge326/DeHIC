import gdal
import numpy as np
import pandas as pd


# Read image
def read_tif_as_frame(imagery_path):
    # Read image
    dataset = gdal.Open(imagery_path)
    dsmatrix = dataset.ReadAsArray(xoff=0, yoff=0, xsize=dataset.RasterXSize, ysize=dataset.RasterYSize)
    
    # Get meta data
    num_bands = dataset.RasterCount
    # Unfold array into pandas DataFrame
    rows = dsmatrix.shape[1]
    cols = dsmatrix.shape[2]
    data_array = dsmatrix[0,:,:]
    for i_band in range(1,num_bands):
        tempmatirx = dsmatrix[i_band,:,:]
        data_array = np.vstack((data_array,tempmatirx))
    data_frame = pd.DataFrame(data_array)
    
    return data_frame, rows, cols, num_bands

# Convert DataFrame to numpy array
def dataframe_to_matrix(z_p_frame, rows, cols, num_bands):

    # Convert the result Frame to Matrix
    result_matrix = np.empty((num_bands, rows, cols))
    for i in range(0,rows):
        result_matrix[i,:] = z_p_frame.loc[i*cols:(i+1)*cols-1].T 
    
    return result_matrix