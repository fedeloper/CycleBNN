
#!/bin/sh

# Navigate to the faster_rcnn/lib directory
cd faster_rcnn/lib

# Run the Python script for setup
/bin/python3 setup.py build develop 

# Navigate back to the original directory
cd ../..

# Execute the training/validation script with specified parameters
#/bin/python3 faster_rcnn/test_net.py --dataset voc --data_root /data1/fontana/BiDet/data/ --basenet /data1/fontana/CycleBNN/pruning/logs/imagenet_resnet_18/dulcet-glitter-408_checkpoint_8947500_60.45.pth.tar

/bin/python3 faster_rcnn/test_net.py --dataset='voc' --checkpoint='/data1/fontana/BiDet/logs/voc/bidet18_IB/2024-02-06 00:38:48/model_20_loss_0.5733_lr_1e-05_rpn_cls_0.1296_rpn_bbox_0.0654_rcnn_cls_0.169_rcnn_bbox_0.1392_rpn_prior_0.0411_rpn_reg_0.0009_head_prior_0.0276_head_reg_0.0006.pth'