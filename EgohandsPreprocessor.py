# File: EgohandsPreprocessor.py
# Author: Nihesh Anderson, Harsh Pathak

"""
This file reads the Egohands dataset, and converts it into a suitable format that can be understood by VGG-SSD implementation

Usage:
python3 EgohandsPreprocessor.py 

Optional Arguments
--nomerge   - The images in different subfolders in the original dataset is not merged into one common train folder  

Output: 
Generates ../Dataset/egohands_data that is compatible with SSD-qfgaohao

"""

import sys
from scipy.io import loadmat
import pandas as pd
import numpy as np
import os
import argparse
import glob
from shutil import copyfile
import random

# Absolute/Relative path to Egohands dataset

ROOT = "./../Dataset/egohands_data"

def TrainTestSplit():

    """
    Splits the entire bunch of data into 80-20 train-test sets

    """

    global ROOT

    SUBROOT = ROOT+"/train/"
    images = os.listdir(ROOT)

    # Setting a random state to ensure the same train test split is generated
    random_state = 10
    random.Random(random_state).shuffle(images)

    train = open(ROOT+"/train_files.txt", 'w')
    val = open(ROOT+"/validation_files.txt", 'w')
    
    # 80-20 split
    trainlist = images[0: int(0.8 * len(images))]
    trainset = set(trainlist)

    vallist = [item for item in images if item not in trainset]

    for i in trainlist:
            train.write(i + "\n")
    for i in vallist:
            val.write(i + "\n")

    train.close()
    val.close()

def CleanEgohandsPackage():
    
    """
    Removes unnecessary files from the downloaded Egohands dataset

    """

    global ROOT

    # Delete unnecessary files from the standard egohands dataset
    os.system("rm "+str(ROOT)+"/*.m*")
    os.system("rm "+str(ROOT)+"/README.txt")
    os.system("rm -rf "+str(ROOT)+"/_LABELLED_SAMPLES")

if(__name__ == "__main__"):

    parser = argparse.ArgumentParser(description = 'Arguments for script')

    parser.add_argument('--nomerge', default = False, action = 'store_true',
                    help = 'if True, folders will be merged to one single train folder')
    
    args = parser.parse_args()

    annotations = loadmat(ROOT+"/metadata.mat")
    # x is numpy array of shape (1, 48)
    
    x = annotations['video']
    NEW_FOLDER = ROOT+"/train/"
    if(os.path.exists(NEW_FOLDER)):
        print("Warning: Train folder already exists. Deleting and regenerating...")
        os.system("rm -rf "+str(NEW_FOLDER))

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
                        if(not args.nomerge):
                            copyfile(cur_imglocation, NEW_FOLDER + "/" + str(file_counter) + ".jpg")
                    else:
                        print("Exiting: original filenames mentioned in metadata.mat not found, meaning this script has already renamed images before")
                        sys.exit()
                    counter += 1
            file_counter += 1                
    df.to_csv("./../Dataset/egohands_data/Annotations.csv", index = False)				
    
    # Split the entire data into train and test set
    TrainTestSplit()

    # Deleted unnecessary files
    CleanEgohandsPackage()
