import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py

class MatDataset(Dataset):

    def __init__(self, gt_mat_file,input_mat_file, scale=1.0):
        with h5py.File(input_mat_file, 'r') as input_mat:
            print('Loading dataset...')
            self.inputs = np.array(input_mat['I_prop_hres_abs'], dtype=np.float32)
        with h5py.File(gt_mat_file, 'r') as gt_mat:
            self.gts = np.array(gt_mat['scaled_data'], dtype=np.float32)
        
        print(f"Shape of input data: {self.inputs.shape}")
        print(f"Shape of gt data: {self.gts.shape}")
        self.gts = np.expand_dims(self.gts, axis=-1) 

    def __len__(self):
        return self.inputs.shape[0]  # Number of samples

    def __getitem__(self, idx):
        input_img = self.inputs[idx]
        gt_img = self.gts[idx]

        # Convert NumPy arrays to PyTorch tensors
        input_img = torch.as_tensor(input_img).float()
        gt_img = torch.as_tensor(gt_img).float()

        # Use permute to change the channel order to (C, H, W)
        gt_img = gt_img.permute(2, 0, 1)  # Assuming the input has shape (H, W, C)

        return {
            'image': input_img,
            'target': gt_img
        }
    
    def _resize(self, img):
        img = Image.fromarray(img)
        return np.asarray(img.resize((int(img.width * self.scale), int(img.height * self.scale))))
    
    def get_single_sample(self, idx):
        """Retrieve a single sample from the dataset."""
        sample = self.__getitem__(idx)  # Reuse the __getitem__ method
        return sample['image'], sample['target'] 