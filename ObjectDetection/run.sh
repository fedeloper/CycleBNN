#!/bin/sh

# Navigate to the faster_rcnn/lib directory
cd faster_rcnn/lib

# Run the Python script for setup
/bin/python3 setup.py build develop 

# Navigate back to the original directory
cd ../..

# Execute the training/validation script with specified parameters
/bin/python3 faster_rcnn/trainval_net.py --dataset voc --data_root /data1/fontana/BiDet/data/ --basenet /data1/fontana/CycleBNN/pruning/logs/imagenet_resnet_18/dulcet-glitter-408_checkpoint_8947500_60.45.pth.tar
