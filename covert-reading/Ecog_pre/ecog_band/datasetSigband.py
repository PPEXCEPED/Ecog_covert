'''
Author: pingping yang
Date: 2024-08-13 04:21:44
Description: create each band Ecog data(after stft)
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
class CustomDatasetSigband(Dataset):
    def __init__(self, HS, path_elec, freq, elec, num_samples, band='None',exclude=False):
        '''
            band: band=='None': return all bands  band=='orthers': return each band
            exclude: exclude==False: permute all bands except specific bands, return all band, exclude==True, permute specific bands and return all bands
        '''
        super().__init__()
        self.data = []
        self.labels = []
        self.exclude = exclude
             
        if isinstance(num_samples,int):
            num_samples=[i for i in range(int(num_samples / 3))]

        for num in num_samples:
            cue_path = os.path.join(path_elec, f'{num}_data_block_cue.npy')
            read_path = os.path.join(path_elec, f'{num}_data_block_read.npy')
            baseline_cue_path = os.path.join(path_elec, f'{num}_baseline_block_cue.npy')
            baseline_read_path = os.path.join(path_elec, f'{num}_baseline_block_read.npy')
            # label_path = os.path.join(path_elec, f'{num}_data_label.npy')

            if os.path.exists(cue_path) and os.path.exists(read_path):
                elec_cue = np.load(cue_path)
                elec_read = np.load(read_path) 
                baseline_cue = np.load(baseline_cue_path)
                baselien_read = np.load(baseline_read_path)
                # elec_label = np.load(label_path)
                # print(len(elec_label))
                # print(elec_cue[0].shape)
                # print(elec_read[0].shape) # [510, 375]
                # print(len(elec_cue)) # 60
                
                len_cue=elec_cue.shape[0]
                len_read=elec_read.shape[0]

                for i in range(len_cue):
                    split_band_cue = elec_cue[i] #(510, 375)
                    split_base_cue = baseline_cue[i]
                    if band != 'None':
                        split_band_cue, split_base_cue = self.extracting(elec_cue[i], split_base_cue, freq,band) #(band_len, 375)
                    elec_cue_band = split_band_cue
                    elec_cue_base = split_base_cue
                    elec_cue_norm = (elec_cue_band - elec_cue_base) / elec_cue_base
                    elec_cue_norm=elec_cue_norm[np.newaxis,:,:] #(1, band_len, 375)
                    # elec_cue_base = elec_cue_base[np.newaxis, :, :]
                    # elec_cue_norm = (elec_cue_band - elec_cue_base) / elec_cue_base
                    self.data.append(elec_cue_norm)
                    self.labels.append(0)
                    # print(elec_cue_band.shape)

                for j in range(len_read):
                    split_band_read = elec_read[j]
                    split_base_read = baselien_read[j]
                    if band != 'None':
                        split_band_read, split_base_read = self.extracting(elec_read[j], split_base_read, freq,band)  
                    elec_read_band = split_band_read
                    elec_read_base = split_base_read
                    elec_read_norm = (elec_read_band - elec_read_base) / elec_read_base
                    elec_read_norm=elec_read_norm[np.newaxis, :, :]
                    # elec_read_base = elec_read_base[np.newaxis, :, :]
                    self.data.append(elec_read_norm)  # Append data
                    self.labels.append(1)
                    # print(elec_read_band.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a tuple of (data, label) and convert data to a PyTorch tensor
        return torch.tensor(abs(self.data[idx]), dtype=torch.float), torch.tensor(self.labels[idx])
    
    def extracting(self, stft_block, baseline_data, freq, band):
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

        indices = np.where((f >= bands[band][0]) & (f < bands[band][1]))[0]


        if self.exclude == True:
            # data_to_shuffle = stft_block[indices, :]
            # np.random.shuffle(data_to_shuffle)
            # stft_block[indices, :] = data_to_shuffle
            # filtered_stft_block = stft_block
            stft_block_filtered = np.delete(stft_block, indices, axis=0)
            baseline_data_filtered = np.delete(baseline_data, indices, axis=0)
            
        else:
            # mask = np.ones(stft_block.shape[0], dtype=bool)
            # mask[indices] = False

            # # 提取和打乱除了 indices 之外的数据
            # data_to_shuffle = stft_block[mask, :]
            # np.random.shuffle(data_to_shuffle)
            # # 将打乱的数据放回原数组
            # stft_block[mask, :] = data_to_shuffle
            # filtered_stft_block = stft_block
            stft_block_filtered = stft_block[indices, :]
            baseline_data_filtered = baseline_data[indices, :]

        return stft_block_filtered, baseline_data_filtered
    