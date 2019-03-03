#!/bin/sh

python3.6 eval_ssd.py --dataset_type wildlife --dataset ./../../Dataset/palm_test --label_file models/voc-model-labels.txt --trained_model models/vgg16-ssd-Epoch-95-Loss-1.7684030264616013.pth

python3.6 draw_eval_results.py ./eval_results/det_test_Hand.txt ./../../Dataset/palm_test/train ./visualisation 0.4
