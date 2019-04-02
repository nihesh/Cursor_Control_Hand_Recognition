#!/bin/sh


python3.6 eval_ssd.py --dataset_type wildlife --dataset ./../../Dataset/egohands_data --label_file models/voc-model-labels.txt --trained_model models/vgg16-ssd-Epoch-95-Loss-1.7684030264616013.pth
