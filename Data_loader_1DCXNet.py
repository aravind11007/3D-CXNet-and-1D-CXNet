#!/usr/bin/env python
# coding: utf-8

# Import required libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy

# Define a custom Dataset class for loading UBFC data
class UBFC_LOADER(Dataset):
    def __init__(self, path):
        """
        Initializes the dataset loader by:
        - Converting the path to a Path object
        - Collecting all data-label file pairs
        """
        self.path = Path(path)
        self.all_file = self.ALL_files(self.path)  # List of tuples (data_path, label_path)

    @staticmethod
    def ALL_files(path):
        """
        This static method builds and returns a list of tuples:
        (data file path, label file path)
        """
        ext = os.listdir(path)  # List subdirectories inside the given path
        data_path = path / ext[0]  # Assumes the first subdirectory contains data files
        label_path = path / ext[1]  # Assumes the second subdirectory contains label files
        
        all_files = []
        for ext in os.listdir(data_path):
            DATA_PATH = data_path / ext  # Full path to the data file
            LABEL_PATH = label_path / ext  # Corresponding label file path
            all_files.append((DATA_PATH, LABEL_PATH))  # Add the pair as a tuple
        
        return all_files

    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.all_file)

    def __getitem__(self, idx):
        """
        Returns the (data, label) pair at the given index.
        Applies normalization and detrending.
        """
        data_path, label_path = self.all_file[idx]  # Get paths for data and label
        
        data = np.load(data_path)  # Load data as numpy array
        label = np.load(label_path)  # Load label (e.g., PPG signal)

        # Detrend the label to remove linear trends
        label = scipy.signal.detrend(label)

        # Normalize data to the range [0, 1]
        data = (data - data.min()) / (data.max() - data.min())

        # Convert data and label to PyTorch float tensors
        data = torch.tensor(data).float()
        label = torch.tensor(label).float()

        return data, label  # Return as a tuple
