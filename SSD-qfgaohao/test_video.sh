#!/bin/sh

python3 run_ssd_live_demo.py vgg16-ssd models/vgg16-ssd-Epoch-95-Loss-1.7684030264616013.pth models/voc-model-labels.txt ./../../Dataset/palm_test/testVideo.mp4
