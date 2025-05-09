#!/usr/bin/env python
# coding: utf-8

# Import required libraries
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pickle

# Define a custom PyTorch Dataset for loading UBFC data
class UBFC_LOADER(Dataset):
    def __init__(self, path_name):
        super().__init__()
        self.path = path_name  # Directory where the .pkl files are stored
        self.files = self.all_files(path_name)  # Get and shuffle list of file paths

    @staticmethod
    def all_files(path_name):
        """
        This static method returns a shuffled list of file paths
        from the directory specified by path_name.
        """
        files = []
        for dir in os.listdir(path_name):
            files.append(os.path.join(path_name, dir))
        np.random.shuffle(files)  # Shuffle files to ensure training randomness
        return files

    def __len__(self):
        """
        Return the total number of data samples.
        """
        return len(self.files)

    def __getitem__(self, index):
        """
        Load and return a single data sample at the given index.
        Each sample includes:
        - 'forehead' (the face region, normalized and reshaped)
        - 'ppg' (the corresponding PPG signal)
        """
        path = self.files[index]  # Get the file path for this index

        # Load the data dictionary from the pickle file
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Extract face data and normalize
        forehead = data['face']
        forehead = forehead / np.max(forehead)  # Normalize to [0, 1]

        # Rearrange dimensions: (T,H, W, C) -> (C, T, H, W)
        forehead = np.transpose(forehead, (3, 0, 1, 2))
        forehead = torch.tensor(forehead).float()  # Convert to float tensor

        # Extract and convert PPG signal to tensor
        ppg = data['ppg']
        ppg = torch.tensor(ppg).float()

        return forehead, ppg  # Return both the input and label
