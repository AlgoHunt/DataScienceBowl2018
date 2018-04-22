"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import random
import math
import re
import time
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import csv
import imageio
import skimage
from config import Config
from skimage.morphology import label
from skimage.feature import canny
from skimage import exposure
from keras.callbacks import Callback
from skimage.morphology import binary_closing, binary_opening, disk, binary_dilation
from scipy.ndimage.morphology import binary_fill_holes
from sklearn.externals import joblib
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.morphology import watershed
from skimage.filters import sobel
from imp import reload
import utils
import model_sep_roi_LH as modellib
import visualize
from model_sep_roi_LH import log
import divide

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/nucleus/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
def train_valid_split(meta, validation_size, valid_category_ids=None):
  meta_train = meta[meta['is_train'] == 1]
  meta_train_split, meta_valid_split = split_on_column(meta_train,
                             column='vgg_features_clusters',
                             test_size=validation_size,
                             random_state=1234,
                             valid_category_ids=valid_category_ids
                             )
  return meta_train_split, meta_valid_split


def split_on_column(meta, column, test_size, random_state=1, valid_category_ids=None):
  if valid_category_ids is None:
    categories = meta[column].unique()
    np.random.seed(random_state)
    valid_category_ids = np.random.choice(categories,
                        int(test_size * len(categories)))
  valid = meta[meta[column].isin(valid_category_ids)].sample(frac=1, random_state=random_state)
  train = meta[~(meta[column].isin(valid_category_ids))].sample(frac=1, random_state=random_state)
  return train, valid

meta = pd.read_csv('./stage1_metadata.csv')

meta_ts = meta[meta['is_train']==0]
meta_train, meta_valid = train_valid_split( meta[meta['is_train']==1],0.2,[0])

############################################################
#  Configurations
############################################################

class NucleusDsbConfig(Config):

  # Give the configuration a recognizable name
  NAME = "res101"
    
  LEARNING_RATE = 1e-2
  
  # If enabled, resizes instance masks to a smaller size to reduce
  # memory load. Recommended when using high-resolution image
  USE_MINI_MASK = True
  MINI_MASK_SHAPE = (28, 28)  # (height, width) of the mini-mask
  
  # Train on 1 GPU and 8 images per GPU. Batch size is GPUs * images/GPU.
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1
  # Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch.
  # typically be equal to the number of samples of your dataset divided by the batch size
  STEPS_PER_EPOCH = 606
  VALIDATION_STEPS = 58

  # Number of classes (including background)
  NUM_CLASSES = 1 + 1  # background + nucleis
  IMAGE_MIN_DIM = 768
  IMAGE_MAX_DIM = 768
  IMAGE_PADDING = True  # currently, the False option is not supported
  RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels, maybe add a 256?
  # The strides of each layer of the FPN Pyramid. These values
  # are based on a Resnet101 backbone.
  BACKBONE_STRIDES = [4, 8, 16, 32, 64]
  # How many anchors per image to use for RPN training
  RPN_TRAIN_ANCHORS_PER_IMAGE = 320 #300
  
  # ROIs kept after non-maximum supression (training and inference)
  POST_NMS_ROIS_TRAINING = 2000
  POST_NMS_ROIS_INFERENCE = 2000
  POOL_SIZE = 7
  MASK_POOL_SIZE = 14
  MASK_SHAPE = [28, 28]
  TRAIN_ROIS_PER_IMAGE = 512
  RPN_NMS_THRESHOLD = 0.7
  MAX_GT_INSTANCES = 256
  DETECTION_MAX_INSTANCES = 400 
  # Minimum probability value to accept a detected instance
  # ROIs below this threshold are skipped
  DETECTION_MIN_CONFIDENCE = 0.7 # may be smaller?
  # Non-maximum suppression threshold for detection
  DETECTION_NMS_THRESHOLD = 0.3 # 0.3
  
  #MEAN_PIXEL = np.array([56.02,54.02,54.26])
  MEAN_PIXEL = np.array([123.7,116.8,103.9])
  #MEAN_PIXEL = np.array([.0,.0,.0])
  # Weight decay regularization
  WEIGHT_DECAY = 0.0001
  SCALES = [768]
  MAIN_SCALE = 0
  BACKBONE = "resnet101"
  GRADIENT_CLIP_NORM = 0.5
  




class NucleusInferenceConfig(DsbConfig):
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1

############################################################
#  Dataset
############################################################
import random



class DsbDataset(utils.Dataset):

  def load_dataset(self, dataset_dir,subset='train'):
    self.add_class("dsb", 1, "nuclei")
    assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]
    
    train_dir = 'stage1_train'
    test_dir = 'stage1_test'
    final_dir = 'stage2_test_final'

    
    if subset == "train":
      directory  = os.path.join(dataset_dir, train_dir)
      
    else subset == "test":
      directory = os.path.join(dataset_dir, test_dir)
    elif subset == "final":
      directory = os.path.join(dataset_dir, final_dir)

    ids = os.list_dir(directory)
    for i, id in enumerate(ids):
      image_dir = os.path.join(directory, id)
      self.add_image("dsb", image_id=i, path=image_dir)
      

  def load_image(self, image_id, non_zero=None):
    info = self.image_info[image_id]
    path = info['path']
    image_name = os.listdir(os.path.join(path, 'images'))
    image_path = os.path.join(path, 'images', image_name[0])
    image = imageio.imread(image_path)
    
    if len(image.shape)==2:
      img = skimage.color.gray2rgb(image)
      image = img*255.
    if image.shape[2] >3:
      image = image[:,:,:3]

    #image = self.preprocess(image)
    image = image
    return image

  def image_reference(self, image_id):
    info = self.image_info[image_id]
    if info["source"] == "shapes":
      return info["shapes"]
    else:
      super(self.__class__).image_reference(self, image_id)

  def load_mask(self, image_id):
    info = self.image_info[image_id]
    path = info['path']
    mask_dir = os.path.join(path, 'masks')
    mask_names = os.listdir(mask_dir)
    count = len(mask_names)
    mask = []
    for i, el in enumerate(mask_names):
      msk_path = os.path.join(mask_dir, el)
      msk = imageio.imread(msk_path)
      if np.sum(msk) == 0:
        print('invalid mask')
        continue
      msk = msk.astype('float32')/255.
      if len(msk.shape) == 3:
        msk = msk[:,:,0]
      mask.append(msk)
    mask = np.asarray(mask)
    mask[mask > 0.] = 1.
    mask = np.transpose(mask, (1,2,0))
    occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
    count = mask.shape[2]
    for i in range(count-2, -1, -1):
      mask[:, :, i] = mask[:, :, i] * occlusion
      occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
    class_ids = [self.class_names.index('nuclei') for s in range(count)]
    class_ids = np.asarray(class_ids)
    return mask, class_ids.astype(np.int32)
  
  def preprocess(self, img):
    gray = skimage.color.rgb2gray(img.astype('uint8'))
    img = skimage.color.gray2rgb(gray)
    img *= 255.
    return img
############################################################
#  Training
############################################################

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = NucleusDataset()
    dataset_train.load_nucleus(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NucleusDataset()
    dataset_val.load_nucleus(args.dataset, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights

        
    print("Train network heads")
    model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=30,
            layers='heads')
        
    print("Train all layers")
    model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE/10,
            epochs=40,
            layers='all')
        

############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = NucleusDataset()
    dataset.load_nucleus(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = NucleusConfig()
    else:
        config = NucleusInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
