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
import argparse
import glob
from shutil import copyfile

# Absolute/Relative path to Egohands dataset

ROOT = "./../Dataset/egohands_data"


if(__name__ == "__main__"):

    parser = argparse.ArgumentParser(description = 'Arguments for script')

    parser.add_argument('--merge', default = False, action = 'store_true',
                    help = 'if True, folders will be merged to one single train folder')
    
    args = parser.parse_args()
    annotations = loadmat(ROOT+"/metadata.mat")
    # x is numpy array of shape (1, 48)
    
    x = annotations['video']
    NEW_FOLDER = "./../Dataset/egohands_data/train/"
    if(os.path.exists(NEW_FOLDER)):
        print("Exiting: new folder already exists, delete before use")
        sys.exit()
    os.makedirs(NEW_FOLDER)    
    df = pd.DataFrame(columns = ['Filename', 'LabelId', 'Label', 'xmin', 'ymin', 'xmax', 'ymax'])
    counter = 0
    file_counter = 0
    l = []

    #fid is iterator as in folder id , since there are 48 folders
    for fid in range(0, x.shape[1]):
            folderarr = x[0, fid][6]
            # folderarr has dimension (1, 100) since there are 100 images in each folder
            # imgid = image id
            for imgid in range(0, folderarr.shape[1]):
                    # temp is an array.void that stores info on boundary pixels of maximum 4 hands in each image
                    temp = folderarr[0, imgid]
                    lenth = len((folderarr[0, imgid]))
                    # iter iterates in numpy arrays, each numpy array describes pixel boundary info of each hand(
                    # max 4 hands)
                    for iter in range(1, lenth):
                            # image name is a number thats at 0 index of temp
                            imgname = str(temp[0][0][0])
                            # make 12 as 0012 and 131 as 0131
                            for ad in range(0, 4 - len(imgname)):
                                    imgname = '0' + imgname
                            #l is useless here
                            l.append(imgname)
                            if(temp[iter].shape[0] != 0):
                                    n = temp[iter]
                                    minxy = (n.min(axis = 0))
                                    maxxy = (n.max(axis = 0))
                                    cur_imglocation = "./../Dataset/egohands_data/_LABELLED_SAMPLES/" + str(x[0, fid][0][0]) + "/frame_" + imgname + ".jpg"
                                    final_imglocation = "./../Dataset/egohands_data/_LABELLED_SAMPLES/" + str(x[0, fid][0][0]) + "/" + str(file_counter) + ".jpg"
                                    df.loc[counter] = [str(file_counter) + ".jpg", 1, "Hand", int(minxy[0]), int(minxy[1]), int(maxxy[0]), int(maxxy[1])]
                                    if(os.path.exists(cur_imglocation)):
                                        if(args.merge):
                                            copyfile(cur_imglocation, NEW_FOLDER + "/" + str(file_counter) + ".jpg")
                                    else:
                                        print("Exiting: original filenames mentioned in metadata.mat not found, meaning this script has already renamed images before")
                                        sys.exit()
                                    counter += 1
                    file_counter += 1                
    df.to_csv("./../Dataset/Annotations.csv", index = False)				
    
