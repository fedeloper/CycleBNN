#!/bin/sh

# Navigate to the faster_rcnn/lib directory
cd faster_rcnn/lib

# Run the Python script for setup
/bin/python3 setup.py build develop 

# Navigate back to the original directory
cd ../..

# Execute the training/validation script with specified parameters
/bin/python3 ../ObjectDetection/faster_rcnn/trainval_net.py --dataset voc --data_root ../ObjectDetection/data/ --basenet "path to weights"
