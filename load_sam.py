import os
import sys
sys.path.append("..")
import numpy as np
import os, json
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader
import cv2
from transforms import ResizeLongestSide
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from config import cfg
class SA1BDataset(Dataset):
    def __init__(self, dataset_dir, annotation_dir='annotations', image_dir='images', min_object=0, transform=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
        self.min_object = min_object
        self.transform = transform
        self.samples = [file.replace(".json",'') for file in os.listdir(os.path.join(dataset_dir, annotation_dir))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        id = self.samples[index]
        image_path = os.path.join(self.dataset_dir, self.image_dir, id + ".jpg")
        annotation_path = os.path.join(self.dataset_dir, self.annotation_dir, id + ".json")

        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if "annotations" in annotation:
            bboxes, masks = self.load_bboxes_mask(annotation["annotations"])
        else:
            bboxes, masks = [], []

        if self.transform:
            image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)
        return image, torch.tensor(bboxes), torch.tensor(masks).float()

    def load_bboxes_mask(self, annotations):
        masks = []
        bboxes = []
        for annotation in annotations:
            masks.append(mask_utils.decode(annotation['segmentation'])) 
            x, y, w, h = annotation['bbox']
            bboxes.append([x, y, x + w, y + h])
        return bboxes, masks
    
class ResizeAndPad:

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes):
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        masks = [transforms.Pad(padding)(mask) for mask in masks]

        # Adjust bounding boxes
        bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
        bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]

        return image, masks, bboxes
def collate_fn(batch):
    images, bboxes, masks = zip(*batch)
    images = torch.stack(images)
    return images, bboxes, masks

from torch.utils.data import random_split

def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    dataset = SA1BDataset(dataset_dir=cfg.samdataset,transform=transform)
    train_size = int(0.8 * len(dataset))  
    val_size = len(dataset) - train_size  
    train, val = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val,
                                batch_size=cfg.batch_size,
                                shuffle=False,
                                num_workers=cfg.num_workers,
                                collate_fn=collate_fn)
    return train_dataloader, val_dataloader

# train_dataloader,_=load_datasets(cfg=cfg,img_size=1024)

# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)
# def show_res(masks, input_box, image):
#     for i,mask in enumerate(masks):
#         plt.figure(figsize=(10,10))
#         plt.imshow(image)
#         show_mask(mask, plt.gca())
#         if input_box is not None:
#             box = input_box[i]
#             show_box(box, plt.gca())
#         plt.axis('off')
#         plt.show()
# if __name__ == '__main__':
#     for images, bboxes, masks in train_dataloader:
#         image = images[0].permute(1, 2, 0).numpy()  
#         bboxes = bboxes[0].numpy()
#         masks = masks[0].numpy()
#         show_res(masks,bboxes,image)
#         break 




