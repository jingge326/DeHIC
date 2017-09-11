#-*-coding:UTF-8-*-
"""
Plotting curves of Hyperspectral data by pixels to faciliate spectral resampling

This not necessary for data process.

"""
# Authors: Jingge Xiao <jingge.xiao@gmail.com>
#
# Created on Fri Jul 28 10:21:50 2017

import numpy as np
import csv
import matplotlib.pyplot as plt

spec_file = r'E:\Research\HyperspectralImageClassification\Experiment\Data\Unlabeled\ing\Hyperion.csv'
img_file = r'E:\Research\HyperspectralImageClassification\Experiment\Data\Unlabeled\ing\npy\1.npy'

point_list = []
csv_reader = csv.reader(open(spec_file, encoding='utf-8'))
for row in csv_reader:
    point_list.append(float(row[0]))

point_array=np.array(point_list)
spec_cube_list = np.load(img_file)
spec_array = spec_cube_list[1, 1, 1, :]
plt.plot(point_array, spec_array)




ip_spec_file = r'E:\Research\HyperspectralImageClassification\Experiment\Data\Unlabeled\ing\IndianPines.csv'
ip_img_file = r'E:\Research\HyperspectralImageClassification\Experiment\xiao\IP_corrected_enlarged.npy'

ip_point_list = []
csv_reader = csv.reader(open(ip_spec_file, encoding='utf-8'))
for row in csv_reader:
    ip_point_list.append(float(row[0]))

ip_point_array=np.array(ip_point_list)
ip_spec_cube_list = np.load(ip_img_file)
ip_spec_array = ip_spec_cube_list[1, 1, 1, :]
plt.plot(ip_point_array, ip_spec_array)

