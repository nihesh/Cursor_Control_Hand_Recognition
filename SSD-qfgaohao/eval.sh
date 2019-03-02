#!/bin/sh


python3.6 eval_ssd.py --dataset_type wildlife --dataset ./../../Dataset/egohands_data --label_file models/voc-model-labels.txt --trained_model models/vgg16-ssd-Epoch-110-Loss-2.3571143956052456.pth
