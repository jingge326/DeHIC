# DeHIC #

<img src="https://github.com/jingge326/MaterialFolder/blob/master/HyperspectralCube.jpg" width = "200" height = "200" alt="" align=center />

## Deep Learning Framework for Hyperspectral Image Classification ##

This project focuses on establishing a framework for hyperspectral image classification using deep learning method. As is well known, supervised deep learning methods are currently state of the art for many machine learning problems, but these methods require large quantities of labeled data to be effective. Unfortunately, existing labeled HSI benchmarks are too small to directly train a deep supervised network. Alternatively, I used self-taught learning, which is an unsupervised method to learn feature extracting frameworks from unlabeled hyperspectral imagery. These models learn how to extract generalizable features by training on sufficiently large quantities of unlabeled data that are distinct from the target data set. Once trained, these models can extract features from smaller labeled target data sets.

In the first stage of the experiment, I used both 2d-convolutional network and 3d-convolutional network as autoencoder to produce encoded features. The networks were pre-trained by Hyperion data, and were applied to the Indian Pines data set. Compared with PCA, the results produced by DeHIC is very promising.

<img src="https://github.com/jingge326/MaterialFolder/blob/master/class_num.png" width = "500" height = "451" alt="" align=center />

Results for the first phase of experiments is shown as Table 2.

<img src="https://github.com/jingge326/MaterialFolder/blob/master/results.png" width = "250" height = "180" alt="" align=center />

The author is working on further experiments, and this repository will be updated as more promising results come out. This introduction will focus on explaining how to process labelled and unlabelled hyperspectral images to make them applicable to current deep learning methods. So users interested in deep learning or hyperspectral remote sensing, can try to find some learning materials on the Internet or contact the author(jingge.xiao@gmail.com).

## Unlabelled Data ##

Unlabelled Hyperion data sets were used to pretrain the networks.

Hyperion is an HSI sensor located on NASA’s EO-1 satellite. Hyperion data contain 242 VNIR/SWIR spectral bands; however, after drop “Bad Bands” and bands seriously affected by the absorption of water vapor, 175 bands are last. The GSD of each HSI data set is 30.5 m, which is larger than that of labeled data sets used in this experiment.

Figure 1 is some examples of used Hyperion data sets

<img src="https://github.com/jingge326/MaterialFolder/blob/master/Hyperion.png" width = "600" height = "318" alt="" align=center />

Patches were constructed from Hyperion data sets(shown in Figure 2) for the training of networks. Each patch is a hyperspectral cube with shape 180\*8\*8(bands\*rows\*columns). The width and height of the cubes were 8 pixels, and the step is also 8 pixels. So there is no overlap between two adjacent patches.

<img src="https://github.com/jingge326/MaterialFolder/blob/master/HyperPatches.png" width = "800" height = "166" alt="" align=center />

Band resampling is needed for both unlabelled and labelled data to make them have same spectral bands which is essential of the pre-train process of networks. 180 bands were left after band resampling.

The total number of Hyperion patches used in the experiment is 189466.

## Labelled Data ##

labelled Indian Pines data set was used to evaluate DeHIC.

Indian Pines(shown in Figure 3) is a 200*145*145 data set that was collected over Northwestern Indiana. The original Indian Pines data set has 224 bands, but bands 104–108, 150–163, and 220–224 were removed due to atmospheric absorption or low signalto-noise ratio (SNR). The ground-truth contains 16 classes of different crops and crop mixtures. Indian Pines has a GSD of 20 m. 

<img src="https://github.com/jingge326/MaterialFolder/blob/master/ip.png" width = "500" height = "214" alt="" align=center />

The way of constructing patches from Indian Pines has two significant differences from that of Hypersion data sets.

1. The step is 1 in Indian Pines while equals to width and height of patches in Hyperion data sets. So each patch in Indian Pines represents the pixel(target pixel) in the centre of the patch.
2. In order to make the patches more representative to their target pixels and to make the patch rotate invariant, I copyed the value of target pixel to pixels on its right, bottom and bottom right(shown in Figure 4).

<img src="https://github.com/jingge326/MaterialFolder/blob/master/center.png" width = "600" height = "214" alt="" align=center />

The total number of Indian Pines patches used in the experiment is 145\*145=21025.

## Functions of Each File ##
- **samples\_select_pos.py**

&emsp;&emsp;Randomly select positions of samples for training and validation 

- **unlabeled\_interp.py**

&emsp;&emsp;Resample of of spectral values

&emsp;&emsp;Hypersion data to pretrain autoencoder network should have the similiar spectral bands
with data set to be classified.

- **unlabeled\_scale.py**

&emsp;&emsp;Normalizing pixel values within each band

&emsp;&emsp;Mean and std are calculated using all unlabelled data

- **unlabeled\_produce_patches.py**

&emsp;&emsp;Splitting original Hyperion data into patches for subsequent network training

- **labeled\_preprocess.py**

&emsp;&emsp;Processing Indian Pines data set. 

&emsp;&emsp;Including:

&emsp;&emsp;Resampling spectral values

&emsp;&emsp;Normalizing pixel values within each band

&emsp;&emsp;Constructing patches for subsequent network training and predicting

- **labeled\_split_2d.py**

&emsp;&emsp;Splitting Indian Pines data set into training and validation sets according to selected positions

&emsp;&emsp;This file is for 2d-convolutional networks.

- **labeled\_split_3d.py**

&emsp;&emsp;Splitting Indian Pines data set into training and validation sets according to selected positions

&emsp;&emsp;This file is for 3d-convolutional networks.

- **cae\_2d_12.py**

&emsp;&emsp;An example of 2d-convolutional autoencoder with 12 encoded features

&emsp;&emsp;The autoencoder network is pretrained by nearly 1.9 million unlabelled hyperion patches. As for the classification of Indian Pines dataset, autoencoder network is trained again by Indian Pines dataset patches without label. Then the encoder part is extracted and used to produce encoded features. Finally, features generated from convolutional  encoder is fed to SVM classifier.

- **cae\_3d_12.py**

&emsp;&emsp;An example of 3d-convolutional autoencoder with 12 encoded features

- **pca\_split.py**

&emsp;&emsp;Basic process of Indian Pines data set for PCA-based classification

&emsp;&emsp;Including:

&emsp;&emsp;Normalizing pixel values within each band

&emsp;&emsp;PCA transformation

&emsp;&emsp;Splitting data into training and validation sets according to selected positions

- **svm\_pca.py**

&emsp;&emsp;Basic implement of PCA classifier

- **curves\_plot.py**

&emsp;&emsp;Plotting curves of Hyperspectral data by pixels to faciliate spectral resampling

&emsp;&emsp;This not necessary for experiments.

- **example\_mnist_1.py**

&emsp;&emsp;An example of using convolutional autoencoder on mnist dataset.

- **example\_mnist_2.py**

&emsp;&emsp;Trains a Stacked What-Where AutoEncoder built on residual blocks on the MNIST dataset.

## In The End ##
Although with the development of machine learning in recent years, natural language processing, target detection, image classification and speech recognition have achieved great accuracy improvement, and many excellent applications have been brought into people's daily lives. The development of remote sensing image classification has not made an outstanding breakthrough, mainly due to the complexity of data, insufficient labelled samples and limited innovative methods. More work is needed in this field.

DeHIC aims at promoting deep learning methods in hyperspectral image classification. It seeks to build a framework that enables deep learning methods to be more accurate and convenient in this field. In the first phase of the experiment, I tried convolutional autoencoder. However, as for the method, there so many others needed to be tried, such as CNN, RNN, LSTM, DBN, etc. The author will try to add some residual blocks and deepen the current autoencoder network to see whether it can achieve better results. Frankly, any suggestion or “commits” to DeHIC on GitHub is welcomed!

