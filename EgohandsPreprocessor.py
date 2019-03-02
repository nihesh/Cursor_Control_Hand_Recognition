# File: EgohandsPreprocessor.py
# Author: Nihesh Anderson, Harsh Pathak

"""
This file reads the Egohands dataset, and converts it into a suitable format that can be understood by VGG-SSD implementation

Usage:
python3 EgohandsPreprocessor.py <path to root directory of egohands dataset>

Output: 
Generates Egohands folder in the cwd

"""

import sys
from scipy.io import loadmat
import pandas as pd
import numpy as np
import os

# Absolute/Relative path to Egohands dataset

ROOT = "./../Dataset/egohands_data"

if(__name__ == "__main__"):

    annotations = loadmat(ROOT+"/metadata.mat")
    x = annotations['video']

    df = pd.DataFrame(columns = ['Filename', 'LabelId', 'Label', 'xmin', 'ymin', 'xmax', 'ymax'])
    counter = 0
    l = []

    # Need some comments here please. Messy code is hard to understand.
    for fid in range(0, x.shape[1]):
            folderarr = x[0, fid][6]
            for imgid in range(0, folderarr.shape[1]):
                    temp = folderarr[0, imgid]
                    lenth = len((folderarr[0, imgid]))
                    for iter in range(1, lenth):
                            imgname = str(temp[0][0][0])
                            for ad in range(0, 4 - len(imgname)):
                                    imgname = '0' + imgname
                            l.append(imgname)
                            if(temp[iter].shape[0] != 0):
                                    n = temp[iter]
                                    minxy = (n.min(axis = 0))
                                    maxxy = (n.max(axis = 0))
                                    df.loc[counter] = [str(imgname) + ".jpg", 1, "Hand", int(minxy[0]), int(minxy[1]), int(maxxy[0]), int(maxxy[1])]
                                    counter += 1
                                    
    df.to_csv("./../Dataset/Annotations.csv", index = False)				

