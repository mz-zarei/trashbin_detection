import argparse
# Basic python and ML Libraries
import os
import random
import numpy as np
import pandas as pd
import json

# helper functions
from myutils import *

# torchvision libraries
import torch
import torchvision

# these are the helper libraries imported.
from engine import train_one_epoch, evaluate
import utils
import transforms as T




parser = argparse.ArgumentParser(description='Training object detection model')
parser.add_argument('--images-dir', type=str, default='./training/data',
                    help='path to image folder')
parser.add_argument('--basemodel', type=str, default='retinanet',
                    help='whether to use RetinaNet or FasterRCNN')
parser.add_argument('--epochs', type=int, default=5,
                    help='the number of training epochs')
parser.add_argument('--step-size', type=int, default=3,
                    help='The step size for lr scheduler')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='after each step size, lr will be reduced by gamma factor') 
parser.add_argument('--seed', type=int, default=1,
                    help='seed number')
parser.add_argument('--batch-size', type=int, default=10,
                    help='batch size')
parser.add_argument('--corr-hight', type=int, default=480,
                    help='hight of the image to be converted')
parser.add_argument('--corr-width', type=int, default=480,
                    help='width of the image to be converted')
parser.add_argument('--num-workers', type=int, default=4,
                    help='the number of processes that generate batches in parallel')
parser.add_argument('--learning-rate', type=float, default=0.005,
                    help='learning rate')            
args = parser.parse_args()

# settings
IMAGES_DIR = args.images_dir
BASEMODEL = args.basemodel
SEEDNUM    = args.seed
BATCH_SIZE = args.batch_size
CORR_HEIGHT = args.corr_hight
CORR_WIDTH = args.corr_width
NUM_WORKER = args.num_workers
LR = args.learning_rate
EPOCHS = args.epochs
STEP_SIZE = args.step_size
GAMMA = args.gamma


MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005


# create the directory ./models 
if not os.path.isdir("./models"):
    os.makedirs("./models")



def main():
    # set the seed numbero
    seed(seednum = SEEDNUM)
    # create label_df and label_map
    labels_df, labels_df_train, labels_df_test, label_map =\
    create_label_df(path_to_labels = "./training/labels.json", seednum=SEEDNUM)
    print("- label_df and label_map are created.")

    # create data loaders
    dataset_train = CostumeImageDataset(IMAGES_DIR, CORR_WIDTH, CORR_HEIGHT, labels_df_train, label_map, transforms = get_transform(train=True))
    dataset_test  = CostumeImageDataset(IMAGES_DIR, CORR_WIDTH, CORR_HEIGHT, labels_df_test , label_map, transforms = get_transform(train=False))
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER,
        collate_fn=utils.collate_fn)
    print("- data loaders are created.")

    # to train on gpu if selected.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = len(label_map.keys())

    # get the model using our helper function
    model = get_object_detection_model(num_classes, basemodel=BASEMODEL)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=STEP_SIZE,
                                               gamma=GAMMA)

    # training
    print("- model training started.")
    for epoch in range(EPOCHS):
        # training for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        # save model
        torch.save(model.state_dict(), f'./models/{BASEMODEL}_e{epoch}.pt')


if __name__ == '__main__':
    main()
