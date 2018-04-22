# DataScienceBowl2018-25th

This is the Solution for data science bowl 2018 challenge

the result for publc LB is 0.458 at 240/3634

the result for private LB is 0.557 at [25/3634](https://www.kaggle.com/bravelucky)(epoch 64) and 0.498 at [111/3634](https://www.kaggle.com/algohunt)(epoch 65), 

Because we have missed the merge deadline we have to make 2 different submit. the huge gap between two epoch may comes from huge data mismatch between stage1 and stage2 dataset, as the two model both provide similar proformance in stage1 LB and local CV.

The code is based on mateerport's mask rcnn：[https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)

The main improvement comes from 
1. better roi align implementation which is modified from [tensorpack faster rcnn](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/FasterRCNN)
2. large separable convolution from[Light Head Rcnn Paper](https://arxiv.org/abs/1711.07264)
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

##  Usage
**The Command line tool hasn't been test yet, if you encounter any bug, see main.ipynb as conference**

Train a new model starting from ImageNet weights using `train` dataset (which is `stage1_train` minus validation set)
```
python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet
```

Train a new model starting from specific weights file using the full `stage1_train` dataset
```
python3 nucleus.py train --dataset=/path/to/dataset --subset=stage1_train --weights=/path/to/weights.h5
```

Resume training a model that you had trained earlier
```
python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last
```

Generate submission file from `stage1_test` images
```
python3 nucleus.py detect --dataset=/path/to/dataset --subset=stage1_test --weights=<last or /path/to/weights.h5>
```
