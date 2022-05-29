# import the necessary packages
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2

class SegmentationDataset(Dataset):
  def __init__(self, imagePaths, maskPaths, itransforms=None, mtransforms=None):
    # store the image and mask filepaths, and augmentation
    # transforms
    self.imagePaths = imagePaths
    self.maskPaths = maskPaths
    self.itransforms = itransforms
    self.mtransforms = mtransforms

  def __len__(self):
    # return the number of total samples contained in the dataset
    return len(self.imagePaths)

  def __getitem__(self, idx):
    # grab the image path from the current index
    imagePath = self.imagePaths[idx]

    # load the image from disk, swap its channels from BGR to RGB,
    # and read the associated mask from disk in grayscale mode
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if self.itransforms is not None:
      image = self.itransforms(image)

    mask = torch.as_tensor(cv2.resize(cv2.imread(self.maskPaths[idx]), (image.shape[2],image.shape[1]), interpolation=cv2.INTER_NEAREST), dtype=torch.uint8)
    mask = torch.permute(mask, (2, 0, 1))

    # check to see if we are applying any transformations
    # apply the transformations to both image and its mask
    if self.mtransforms is not None:
      mask = self.mtransforms(mask)

    # SUIM: extract binary mask from multi-label mask (BGR)=(011) (Fish class)
    mask = ((torch.logical_not(mask[:][0])) & (mask[:][1]).to(bool) & (mask[:][2]).to(bool)).to(torch.float32)
    mask.unsqueeze_(0)

    # return a tuple of the image and its mask
    return (image, mask)
