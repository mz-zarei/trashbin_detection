# Basic python and ML Libraries
import os
import random
import numpy as np
import pandas as pd
import json
# We will be reading images using OpenCV
import cv2
# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans  
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights


# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# train-test spli
from sklearn.model_selection import train_test_split
# these are the helper libraries imported.
from engine import train_one_epoch, evaluate
import utils
import transforms as T
# for image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2




class CostumeImageDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading and processing images for object detection tasks.
    Args:
        images_dir (str): Directory containing the images.
        width (int): Width to resize the images to.
        height (int): Height to resize the images to.
        labels_df (pandas DataFrame): DataFrame containing the labels for each image.
        label_map (dict): Mapping of label names to label IDs.
        transforms (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, images_dir, width, height, labels_df, label_map, transforms=None):
        self.transforms = transforms
        self.files_dir = images_dir
        self.height = height
        self.width = width
        self.imgs = [image_name for image_name in labels_df.file_name.values]
        # classes: 0 index is reserved for background
        self.classes = ["_"] + list(label_map.values())
        self.labels_df = labels_df

    def __getitem__(self, idx):
        """
        Returns the image and label information for the specified index.
        Args:
            idx (int): Index of the image to retrieve.
        Returns:
            tuple: Tuple containing the image and label information.
        """
        img_name = self.imgs[idx]
        img_path = os.path.join(self.files_dir, img_name)
        img_info = self.labels_df[self.labels_df.file_name == img_name].reset_index(drop=True)

        # reading the images and converting them to correct size and color    
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0
        

        boxes = []
        labels = []

        # cv2 image gives size as height x width
        wt = img.shape[1]
        ht = img.shape[0]
        
        # box coordinates are extracted and corrected for image size given
        for box, label in zip(img_info.bbox.values, img_info.label_id.values):
            labels.append(label)
            
            # bounding box
            xmin = int(box[0])
            ymin = int(box[1])
            
            xmax = xmin + int(box[2])
            ymax = ymin + int(box[3])

            xmin_corr = (xmin/wt)*self.width
            xmax_corr = (xmax/wt)*self.width
            ymin_corr = (ymin/ht)*self.height
            ymax_corr = (ymax/ht)*self.height
            
            boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])
        
        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        # image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id


        if self.transforms:
            sample = self.transforms(image = img_res,
                                     bboxes = target['boxes'],
                                     labels = labels)
            
            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])       
        
        return img_res, target

    def __len__(self):
        return len(self.imgs)

def create_label_df(path_to_labels="./training/labels.json", seednum=42):
    """
    Returns label dataframes for train and test, and label map based on label.json file.
    Args:
        path_to_labels (str, optional): Path to label JSON file.
        seednum (int, optional): Seed number for train-test split.
    Returns:
        tuple: Tuple containing:
            - labels_df (pd.DataFrame): DataFrame of all labels.
            - labels_df_train (pd.DataFrame): DataFrame of train labels.
            - labels_df_test (pd.DataFrame): DataFrame of test labels.
            - label_map (dict): Dictionary mapping label IDs to label names.
    """
    # Load labels from JSON file
    with open(path_to_labels, 'r') as f:
        labels = json.load(f)
    # Create dataframes from annotations and images lists
    annot_df = pd.DataFrame(labels['annotations'])
    image_df = pd.DataFrame(labels['images'])
    # Create label map dictionary
    label_map = {x['id']: x['name'] for x in labels['categories']}
    # Merge annotation and image dataframes, rename and select columns
    labels_df = annot_df.merge(image_df, how='left', left_on='image_id', right_on='id')
    labels_df['label'] = labels_df['category_id'].map(label_map)
    labels_df = labels_df[['file_name', 'bbox', 'area', 'iscrowd', 'label', 'category_id']].rename({'category_id': 'label_id'}, axis=1)
    # Perform train-test split
    labels_df_train, labels_df_test = train_test_split(labels_df, test_size=0.25, random_state=seednum, shuffle=True)
    return labels_df, labels_df_train, labels_df_test, label_map


def plot_img_bbox(img, target, pred=None, figsize=(12,6)):
    """
    Plots an image with bounding boxes and labels. Optionally plots a second set of bounding boxes and labels representing predictions.
    Args:
        img (np.ndarray): Image to plot.
        target (dict): Dictionary containing the expected bounding boxes and labels.
        pred (dict, optional): Dictionary containing the predicted bounding boxes and labels.
        figsize (tuple, optional): Figure size.
    """
    # Create figure, axes and Plot image
    if pred:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].imshow(img)
        ax[1].imshow(img)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)

    
    # Iterate over bounding boxes and labels
    for ax_idx, data in enumerate([target, pred]):
        if data is None:
            continue
        for box, label in zip(data['boxes'], data['labels']):
            x, y, width, height = box[0], box[1], box[2]-box[0], box[3]-box[1]
            # Create rectangle patch
            rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
            # Draw rectangle on image, Add label
            if pred:
                ax[ax_idx].add_patch(rect)
                ax[ax_idx].text(x, y-12, f'{label}', fontsize=10, c='white')
            else:
                ax.add_patch(rect)
                ax.text(x, y-12, f'{label}', fontsize=10, c='white')
        if pred:
            ax[0].set_title('Expected')
            ax[1].set_title('Predicted')
        else:
            ax.set_title('Expected')
    plt.show()





def get_object_detection_model(num_classes, basemodel='retinanet'):
    """
    Returns a pre-trained object detection model from torchvision.
    Args:
        num_classes (int): Number of classes for the model to classify.
        basemodel (str, optional): Base model to use. Can be either 'retinanet' or 'frcnn'. Default is 'retinanet'.
    Returns:
        model (torchvision.models.detection): Object detection model.
    """
    model = None
    if basemodel == 'retinanet':
        # load pre-trained RetinaNet on COCO
        model = torchvision.models.detection.retinanet_resnet50_fpn(num_classes=num_classes) 
    if basemodel == 'frcnn':
        # load pre-trained FCNN on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes)
    return model

    
def get_transform(train):
    """
    Returns a data transformation for the given dataset.
    Args:
        train (bool): Whether the transformation is for the training set (True) or the test set (False).
    Returns:
        transform (albumentations.core.composition.Compose): Data transformation.
    """
    if train:
        return A.Compose([
                            A.HorizontalFlip(0.5),
                     # ToTensorV2 converts image to pytorch tensor without div by 255
                            ToTensorV2(p=1.0) 
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
                            ToTensorV2(p=1.0)
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def seed(seednum = 1):
    """
    Seeds all relevant random number generators.
    Args:
        seednum (int, optional): Seed number to use. Default is 1.
    """
    random.seed(seednum)
    np.random.seed(seednum)
    torch.manual_seed(seednum)
    torch.cuda.manual_seed(seednum)
    os.environ['PYTHONHASHSEED'] = str(seednum)
    torch.backends.cudnn.deterministic = True



def filter_prediction(orig_prediction, iou_thresh=0.3, score_thresh=0.8):
    """
    Applies non-maximum suppression to the given prediction and filter low scored predictions
    Args:
        prediction (dict): Dictionary containing predicted bounding boxes, labels, and scores.
        iou_threshold (float, optional): Intersection over union threshold to use for non-maximum suppression. Default is 0.3.
        score_threshold (float, optional): Minimum score threshold to use for filtering predictions. Default is 0.8.
    Returns:
        filtered_prediction (dict): Dictionary containing filtered predicted bounding boxes, labels, and scores.
    """
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    filtered_prediction = orig_prediction
    filtered_prediction['boxes'] = filtered_prediction['boxes'][keep].cpu()
    filtered_prediction['scores'] = filtered_prediction['scores'][keep].cpu()
    filtered_prediction['labels'] = filtered_prediction['labels'][keep].cpu()

    # filter prediction with lower score than threshold
    boxes, scores, labels = [], [], []
    for box, label, score in zip(filtered_prediction['boxes'], filtered_prediction['labels'], filtered_prediction['scores']):
        if score >= score_thresh:
            boxes.append(box)
            labels.append(label)
            scores.append(score)
    filtered_prediction = {}
    filtered_prediction['boxes'] = boxes
    filtered_prediction['labels'] = labels
    filtered_prediction['scores'] = scores

    return filtered_prediction

def torch_to_pil(img):
    '''function to convert a torchtensor back to PIL image'''
    return torchtrans.ToPILImage()(img).convert('RGB')








    


