# DataScienceBowl2018-50th

This is the Solution for data science bowl 2018 challenge

the result for publc LB is 0.458 at 240/3634

the result for private LB is 0.557 at [50/3634](https://www.kaggle.com/bravelucky)(epoch 64) and 0.498 at [180/3634](https://www.kaggle.com/algohunt)(epoch 65), 

Because we have missed the merge deadline we have to make 2 different submit. the huge gap between two epoch may comes from huge data mismatch between stage1 and stage2 dataset, as the two model both provide similar proformance in stage1 LB and local CV.

The code is based on mateerport's mask rcnn：[https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)

The main improvement comes from 
1. better roi align implementation which is modified from [tensorpack faster rcnn](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/FasterRCNN)
2. strong image augmentation especially random scale crop
3. using clustering to select proper CV set 
4. divide large picture into small part during inference 
5. finetuning the trainging schedule and mask rcnn configuration

## Dependencies
+ Python 3; TensorFlow >= 1.5.0;keras>=2.0.5
+ Pre-trained [ResNet model](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5) from keras pretrain model .
+ data science bowl data. It assumes the following directory structure:
```
DIR/
  stage1_train/
    images/
    masks/
  stage1_test/
    images/
  stage2_test/
    images/
```
