import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import h5py
import numpy as np


# Building the SeismicDataset class
class SeismicDataset:
    # __init__ runs once you create the dataset. Here you load, normalize and store data. Is the setup step.
    def __init__(self, eq_hdf5, eq_csv, noise_hdf5, noise_csv, max_samples=None):
        super().__init__()
        eq_df = pd.read_csv(eq_csv)
        eq_df = eq_df[eq_df.trace_category == "earthquake_local"]
        if max_samples:
            eq_df = eq_df.iloc[:max_samples]

        # Noise
        noise_df = pd.read_csv(noise_csv)
        noise_df = noise_df[noise_df.trace_category == "noise"]
        if max_samples:
            noise_df = noise_df.iloc[:max_samples]

        eq_waveforms = []
        wf_f = h5py.File(eq_hdf5, "r")
        for name in eq_df.trace_name:
            data = np.array(wf_f[f"data/{name}"])
            eq_waveforms.append(data)
        eq_waveforms = np.array(eq_waveforms)

        noise_waveforms = []
        noise_f = h5py.File(noise_hdf5, "r")
        for name in noise_df.trace_name:
            data = np.array(noise_f[f"data/{name}"])
            noise_waveforms.append(data)
        noise_waveforms = np.array(noise_waveforms)

        # Combine into one array
        waveforms = np.concatenate([eq_waveforms, noise_waveforms], axis=0)
        labels = np.concatenate(
            [np.ones(len(eq_waveforms)), np.zeros(len(noise_waveforms))]
        )

        # Normalization
        waveforms = waveforms / np.max(np.abs(waveforms), axis=(1, 2), keepdims=True)

        # Shuffle
        idx = np.random.permutation(len(waveforms))
        waveforms = waveforms[idx]
        labels = labels[idx]

        # Convert to tensors and store
        # PyTorch needs tensors, not numpy arrays.
        # Also remember the shape needs to go from (N, 6000, 3) to (N, 3, 6000):
        self.X = torch.tensor(waveforms, dtype=torch.float32).permute(
            0, 2, 1
        )  # (N, 3, 6000)
        self.y = torch.tensor(labels, dtype=torch.float32)

        pass

    # __len__ returns the total number of samples.
    def __len__(self):
        return len(self.y)

    # __getitem__ returns one sample given an index.
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
