import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log   # print message

# Root folder of the project
ROOT_DIR = os.getcwd()

# Folder to save log and trained model
MODEL_DIR = os.path.join( ROOT_DIR , "logs")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR , "mask_rcnn_coco.h5")

###################################
# specific configuration of face parsing
class FaceConfig(Config):
    # Give the configuration a recognizable name
    NAME = "face parsing"

    # Train on 1 GPU
    # 2 images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes 
    NUM_CLASSES = 12 # 11 + 1

    # set the limits of the small side , the large side , and that determines the image shape
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640

    # use smaller anchors because our image and objects are small
    # each scale is associated with a level of the pyramid
    # RPN_ANCHOR_SCALES = ( 32 , 64 , 128 , 256 , 512)

    # Reduce training ROIs per image because the images are small and
    # have few objects. Aim to allow ROI sampling to pick 33% positive ROIs
    TRAIN_ROIS_PER_IMAGE = 128

    # use a small epoch since the data is simple
    STEPS_PER_EPOCH = 10

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

config = FaceConfig()
config.display()

######################################
# specific configuration of dataset
class FaceDataset(utils.Dataset):
    def load_faces(self , dataset_dir):
        # add classes
        self.add_class('Face' , 1 , 'body')
        self.add_class('Face' , 2 , 'face')
        self.add_class('Face' , 3 , 'hair')
        self.add_class('Face' , 4 , 'left brow')
        self.add_class('Face' , 5 , 'right brow')
        self.add_class('Face' , 6 , 'left eye')
        self.add_class('Face' , 7 , 'right eye')
        self.add_class('Face' , 8 , 'nose')
        self.add_class('Face' , 9 , 'mouth')
        self.add_class('Face' , 10 , 'left ear')
        self.add_class('Face' , 11 , 'right ear')

        # add image
        # 

    def load_mask(self , image_id):
        pass

    def ToMask(self , ann , height ,width):
        pass

####################################
# Faces Evaluation

def build_faces_results(dataset , image_id , rois , class_ids , scores , masks):
    pass

def evaluate_faces(model ,dataset , faces , eval_type = "bbox" , limit = 0 , image_ids = None):
    pass


####################################
# Training

if __name__  == '__main__':
    import argparse

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN.')
    parser.add_argument('command',
                        metavar = "<command>",
                        help = "'train' or 'evaluate'")
    parser.add_argument('--dataset' , required = True,
                        metavar = "/data",
                        help = "Folder of the dataset")
    parser.add_argument('--model',required = True,
                        metavar = "/mask_rcnn_coco.h5",
                        help = "path to weights .h file or 'coco'")
    parser.add_argument('--logs',required = False ,
                        default = DEFAULT_LOGS_DIR,
                        metavar = "/logs",
                        help = "logs and checkpoints directory")
    parser.add_argument('--limit' , required = False ,
                        default = 500,
                        metavar = "<image counts>",
                        help = "images to use for evaluation")
    args = parser.parse_args()
    print("Command:" , args.command)
    print("Model:" , args.model)
    print("Dataset:" , args.dataset)
    print("Logs:" , args.logs)

    # Configurations
    if args.command == "train":
        config = FaceConfig()
    else:
        class InferenceConfig(FaceConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode = "training" , config = config , 
                                model_dir = args.logs)
    else:
        model = modellib.MaskRCNN(mode = "inference" , config = config,
                                model_dir = args.logs)
    
    # Select weights file to load
    if args.model.lower() == "last":
        model_path = model.find_last()[1]
    elif:
        model_path = args.model

    # Load weights
    print("Loading weights" , model_path)
    model.load_weights(model_path , by_name = True)

    # Train or evaluate
    if args.command == "train":
        # training dataset
        dataset_train = FaceDataset()
        dataset_train.load_faces(args.dataset , "train")
        dataset_train.prepare()
        # validation dataset
        dataset_val = FaceDataset()
        dataset_val.load_faces(args.dataset, "val")
        dataset_val.prepare()

        # Trainoing - Stage 1
        print("Training network heads")
        model.train(dataset_train , dataset_val,
                    learning_rate = config.LEARNING_RATE,
                    epochs = 40,
                    layers = "heads")
        
        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Trianing Resnet layer 4+")
        model.train(dataset_train , dataset_val,
                    learning_rate = config.LEARNING_RATE / 10,
                    epochs = 100,
                    layers = '4+')
        
        # Training - Stage 3
        # Finetune Layers from ResNet stage 3 and up
        print("Training Resnet layer 3+")
        model.train(dataset_train , dataset_val,
                    learning_rate = config.LEARNING_RATE / 100,
                    epochs = 200,
                    layers = 'all')
        
    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = FaceDataset()
        faces = dataset_val.load_faces()
    
    else:
        print("'{}' is not recognized.
                "use 'train' or 'evaluate'".format(args.command))



