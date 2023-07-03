# Codes adapted from MONAI

# Citation:
# Cardoso, M. Jorge, et al.
# "Monai: An open-source framework for deep learning in healthcare."
# arXiv preprint arXiv:2211.02701 (2022).
# Open-Source code: https://github.com/Project-MONAI/MONAI/tree/dev

import numpy as np
import torch
import h5py
import pydicom as dicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union
import helpers
import json
import collections.abc
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from torchvision import transforms
from torchvision.transforms import ToTensor


class HDF5Dataset(Dataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """
    def __init__(self, data: Sequence, transform: Optional[Callable] = None) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.

        """
        self.data = data
        self.transform: Any = transform

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data[index]['image']
        imarray = np.array(h5py.File(data_i, 'r')['data'])
        # Normalization of the image array to [0,1]
        normimarray = (imarray - np.min(imarray)) / (np.max(imarray) - np.min(imarray))
        # Convert to PIL image for transform
        PILimage = Image.fromarray((normimarray * 255).astype(np.uint8))
        # Transform the image
        if self.transform is not None:
            PILimage = self.transform(PILimage)
#         return apply_transform(self.transform, data_i) if self.transform is not None else data_i
        return PILimage

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index)


class DicomDataset(Dataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(self, data: Sequence, transform: Optional[Callable] = None) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.

        """
        self.data = data
        self.transform: Any = transform


    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data[index]['image']
        ds = dicom.dcmread(data_i)
        imarray = apply_voi_lut(ds.pixel_array, ds)
        normimarray = (imarray - np.min(imarray)) / (np.max(imarray) - np.min(imarray))
        PILimage = Image.fromarray((normimarray * 255).astype(np.uint8))
        if self.transform is not None:
            PILimage = self.transform(PILimage)
        return PILimage

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index)


def get_dataloader(jsonfile, batch_size, num_workers):
    # load the dataset from json file
    f = open(jsonfile)
    data = json.load(f)
    trainData = data['training']
    valData = data['test']

    # define Dataset
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_ds = DicomDataset(data=trainData, transform=train_transforms)
    val_ds = DicomDataset(data=valData, transform=val_transforms)
    print('Number of training images: ', len(train_ds))
    print('Number of validation images: ', len(val_ds))

    # DataLoader
    trainDataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    valDataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, drop_last=True)

    return trainDataloader, valDataloader
