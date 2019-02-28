# File: EgohandsPreprocessor.py
# Author: Nihesh Anderson

"""
This file reads the Egohands dataset, and converts it into a suitable format that can be understood by VGG-SSD implementation

Usage:
python3 EgohandsPreprocessor.py <path to root directory of egohands dataset>

Output: 
Generates Egohands folder in the cwd

"""

import sys
from scipy.io import loadmat

# Absolute/Relative path to Egohands dataset
ROOT = None

if(__name__ == "__main__"):

	ROOT = "./../Dataset/egohands_data"

	annotations = loadmat(ROOT+"/metadata.mat")
	print(annotations["video"][0][0])



