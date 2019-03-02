#!/bin/sh

python3.6 ssd_wildlife_detection.py --validation_epochs 3 --datasets ./../../Dataset/egohands_data --validation_dataset ./../../Dataset/egohands_data --net vgg16-ssd --base_net models/vgg16_reducedfc.pth --batch_size 12 --num_epochs 300 --lr 5e-5 --scheduler "multi-step" --milestones "120,160"

