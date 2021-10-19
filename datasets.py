import os
import random

import cv2 as cv
import numpy as np
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset

import torch
import pdb

from tqdm import tqdm
from torchvision.utils import save_image
import torch.nn.functional as F




def get_image_batches(data, labels, set_size, bs, num_class, seed):

    np.random.seed(seed)

    image = torch.zeros([set_size*bs, 3, 224, 224])

    select_label = np.random.permutation(num_class)[0:bs]+1

    
    selected_idx = np.array([])

    for j in range(bs):
        index = np.where(select_label[j]==labels)[0]
        index = index[np.random.permutation(len(index))]

        for i in range(set_size):

            # pdb.set_trace()

            image[set_size*j+i] =  data.dataset[index[i]]

            selected_idx = np.append(selected_idx, int(index[i]))



    return image, select_label-1, selected_idx





