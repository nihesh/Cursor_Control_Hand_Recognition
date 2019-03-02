import sys
from scipy.io import loadmat
import pandas as pd
import numpy as np
import os
import argparse
import glob
import random
ROOT = "./../Dataset/egohands_data/train/"
images = os.listdir(ROOT)

random_state = 10
random.Random(random_state).shuffle(images)
train = open("./../Dataset/egohands_data/train_files.txt", 'w')
val = open("./../Dataset/egohands_data/validation_files.txt", 'w')
print(len(images))
trainlist = images[0: int(0.8 * len(images))]
print(len(trainlist))
set = set(trainlist)

vallist = [item for item in images if item not in set]
print(len(vallist))
for i in trainlist:
	train.write(i + "\n")
for i in vallist:
	val.write(i + "\n")

train.close()
val.close()