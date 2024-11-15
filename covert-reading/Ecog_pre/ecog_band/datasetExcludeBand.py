'''
Author: pingping yang
Date: 2024-08-14 14:48:44
Description: create exclude band Ecog data(after stft)
'''
import numpy as np
import scipy.io as scio
from scipy import signal
import tdt
import os
import  wave
import matplotlib.pyplot as plt
import json
from scipy.fftpack import fft
from random import shuffle
import h5py
import scipy.io as scio
import scipy.io.wavfile
import math
import mne
import os
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils import data as Data
import torch.nn as nn
import seaborn as sns
from ecog_band import EcogBandRes
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset,Dataset
from ecog_band.solver import Nfold_solver
from ecog_band.models import ECOGRes50_feature,ECOGRes50

#can load each band
class CustomDatasetExcband(Dataset):
    def __init__(self, HS, path_elec, freq, elec, num_samples, band):
        super().__init__()
        self.data = []
        self.labels = []
             
        if isinstance(num_samples,int):
            num_samples=[i for i in range(int(num_samples / 2))]

        for num in num_samples:
            cue_path = os.path.join(path_elec, f'elec{elec}_{num}_data_block_cue.npy')
            read_path = os.path.join(path_elec, f'elec{elec}_{num}_data_block_read.npy')

            if os.path.exists(cue_path) and os.path.exists(read_path):
                elec_cue = np.load(cue_path)
                elec_read = np.load(read_path) 
                
                len_cue=elec_cue.shape[0]
                len_read=elec_read.shape[0]

                for i in range(len_cue):
                    elec_cue_band=self.extracting(elec_cue[i],freq,band)
                    # elec_cue_band=elec_cue_band[np.newaxis,:,:]
                    self.data.append(elec_cue_band)
                    self.labels.append(0)

                for j in range(len_read):  
                    elec_read_band=self.extracting(elec_read[j],freq,band)
                    # elec_read_band=elec_read_band[np.newaxis,:,:]
                    self.data.append(elec_read_band)  # Append data
                    self.labels.append(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a tuple of (data, label) and convert data to a PyTorch tensor
        return torch.tensor(abs(self.data[idx]), dtype=torch.float), torch.tensor(self.labels[idx])
    
    def exclude(self,stft_block,freq,band):
        f=torch.arange(stft_block.shape[0])

        bands = {
            'else1': (0,1),
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 70),
            'high gamma':(70,150),
            'else2':(150,freq+1)
        }

        # band_data=np.zeros_like(stft_block)
        indices = np.where((f >= bands[band][0]) & (f < bands[band][1]))[0]
        filtered_stft_block = np.delete(stft_block, indices, axis=0)
        
        return filtered_stft_block
    