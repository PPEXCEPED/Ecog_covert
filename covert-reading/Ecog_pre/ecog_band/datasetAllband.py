'''
Author: pingping yang
Date: 2024-08-13 13:40:58
Description: create dataset in all frequency bands
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
from ecog_band.utils import normalise_stft_data
from sklearn.preprocessing import StandardScaler


#指定频段
class SVMDataset(Data.Dataset):
    def __init__(self, HS, path_elec, freq, elec, num_samples, band='None', exclude=False, avg=None):
        super().__init__()
        self.data = [] # 存Ecog数据，shape为（n_samples x n_features)
        self.labels = [] # 存label, shape=（n_samples），cue标记为0，read标记为1
        self.exclude = exclude # excluding=True delete specific band，excluding=False，return specific band 

        if isinstance(num_samples,int):
            num_samples=[i for i in range(int(num_samples / 5))] # 除4是因为HS69每个电极分6个块的数据 有30的npy文件是因为cue/read/basecue/baseread/baselabel各6个块
        # print(num_samples)
        X_cue,X_read=[],[]
        y_cue,y_read=[],[]
        x_base_cue, x_base_read = [], []
        # print(num_samples)
        for num in num_samples: # num为块的个数

            cue_path = os.path.join(path_elec, f'{num}_data_block_cue.npy')
            read_path = os.path.join(path_elec, f'{num}_data_block_read.npy')
            baseline_cue_path = os.path.join(path_elec, f'{num}_baseline_block_cue.npy')
            baseline_read_path = os.path.join(path_elec, f'{num}_baseline_block_read.npy')
            # print(cue_path)

            if os.path.exists(cue_path) and os.path.exists(read_path):
                elec_cue = np.load(cue_path) # (n_task, n_freq, n_timePoint) (60, 501, 375)
                elec_read = np.load(read_path)[:,:,:elec_cue.shape[2]]
                baseline_cue = np.load(baseline_cue_path)[:, :, 275:]
                baseline_read = np.load(baseline_read_path)
                len_cue = len(elec_cue)
                len_read = len(elec_read)

                # exclude band or choose band
                if band != 'None':
                    exclu_cue = []
                    exclu_read = []
                    base_cue = []
                    base_read = []
                    for i in range(len_cue):
                        filter_cue, filter_base_cue = self.excluding(elec_cue[i], baseline_cue[i], freq, band)
                        exclu_cue.append(filter_cue)
                        base_cue.append(filter_base_cue)
                    for j in range(len_read):
                        filter_read, filter_base_read = self.excluding(elec_read[j], baseline_read[j], freq, band)
                        exclu_read.append(filter_read)
                        base_read.append(filter_base_read)
                        # exclu_read.append(self.excluding(elec_read[i], baseline_read, freq, band))
                    # 对每个block单独做z-score
                    z_base_cue = self.z_score_standardize(exclu_cue, base_cue)
                    z_base_read = self.z_score_standardize(exclu_read, base_cue)
                    X_cue.append(z_base_cue)
                    X_read.append(z_base_read)
                    x_base_cue.append(base_cue) # 每个block取每个看的onset的前0.2s的数据 拼接在一起作为baseline
                    # x_base_read.append(base_read[:, :, :100])
                else:
                    z_x_cue = self.z_score_standardize(elec_cue, baseline_cue)
                    z_x_read = self.z_score_standardize(elec_read, baseline_cue)
                    X_cue.append(z_x_cue)
                    X_read.append(z_x_read)
                    x_base_cue.append(baseline_cue)
                    # print(x_base_cue.shape)
                    # x_base_cue.append(baseline_cue)
                    # x_base_read.append(baseline_read)
                
        
        X_cue=np.abs(np.vstack(X_cue))
        X_read=np.abs(np.vstack(X_read))
        X_base_cue = np.abs(np.vstack(x_base_cue))
        # X_base_read = np.abs(np.vstack(x_base_read))

        # X_cue_norm = (X_cue - X_base_cue) / X_base_cue
        # # X_read_norm = (X_read - X_base_read) / X_base_read
        # X_read_norm = X_cue_norm
        
        # print(f'X_cue.shape: {X_cue.shape}') # (360, 461, 375)
        # print(f'X_base_cue.shape: {X_base_cue.shape}') # (360, 461, 375)

        # z-score
        # X_cue_norm = self.z_score_standardize(X_cue, X_base_cue)
        # X_read_norm = self.z_score_standardize(X_read, X_base_cue)

        # Compute the mean across the frequency dimension (axis=1)
        if avg == 'avgFreq':
            X_cue_mean = np.mean(X_cue, axis=1)
            X_read_mean = np.mean(X_read, axis=1)
            X=np.concatenate((X_cue_mean, X_read_mean),axis=0)
        # print(f'X_cue_mean.shape: {X_cue_mean.shape}') # (720, 375)

        # Compute the mean across the timesteps dimension (axis=1)
        elif avg == 'avgTime':
            X_cue_mean = np.mean(X_cue_norm, axis=2)
            X_read_mean = np.mean(X_read_norm, axis=2)
            X=np.concatenate((X_cue_mean, X_read_mean),axis=0)

        elif avg == 'norm':
            X_cue_norm = self.z_score_standardize(X_cue, X_cue+X_read)
            X_read_norm = self.z_score_standardize(X_read, X_cue+X_read)
            # X_cue_norm = (X_cue - X_base_cue) / X_base_cue
            # X_read_norm = X_cue_norm
            X=np.concatenate((X_cue_norm, X_read_norm),axis=0)
            X = X.reshape(-1, X.shape[1]*X.shape[2])

        elif avg == 'z-score':
            # X_cue_norm = self.z_score_standardize(X_cue, X_base_cue)
            # X_read_norm = self.z_score_standardize(X_read, X_base_cue)
            # self.X_cue_norm = X_cue_norm
            # self.X_read_norm = X_read_norm
            X=np.concatenate((X_cue, X_read),axis=0)
            X = X.reshape(-1, X.shape[1]*X.shape[2])

        else:
            scaler = StandardScaler()
            X=np.concatenate((X_cue, X_read),axis=0)
            X = X.reshape(-1, X.shape[1]*X.shape[2])
            # X = scaler.fit_transform(X)
        # print(X.shape)
        y_cue=np.array([0]*X_cue.shape[0])
        y_read=np.array([1]*X_read.shape[0])
        y=np.concatenate((y_cue,y_read),axis=0)

        self.data = X
        self.labels = y
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a tuple of (data, label) and convert data to a PyTorch tensor
        # print(self.data.shape)
        return torch.tensor(abs(self.data[idx]), dtype=torch.float), torch.tensor(self.labels[idx])
    
    def get_data_labels(self):
        return self.data, self.labels
    
    def get_norm_data(self):
        return self.X_cue_norm, self.X_read_norm
    
    def z_score_standardize(self, X, baseline):
        """计算 z-score 标准化"""
        # 计算拼接后的均值和标准差
        mean_baseline = np.mean(baseline, axis=(0, 2))  # 在样本和时间维度上计算均值
        std_baseline = np.std(baseline, axis=(0, 2))    # 在样本和时间维度上计算标准差

        # 扩展均值和标准差的维度以便与 X 进行广播
        mean_baseline = mean_baseline[np.newaxis, :, np.newaxis]  # 形状变为 (1, n_frequencies, 1)
        std_baseline = std_baseline[np.newaxis, :, np.newaxis]    # 形状变为 (1, n_frequencies, 1)

        # 进行 z-score 标准化
        return (X - mean_baseline) / std_baseline
    
    def expanded_data(self, data, expand_size):
        data_0 = data[0]
        expanded_data_0 = np.expand_dims(data_0, axis=0)
        expanded_data = np.tile(expanded_data_0, (expand_size, 1, 1))
        return expanded_data

    def excluding(self,stft_block, baseline_data, freq, band):
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

        # delete specific band
        indices = np.where((f >= bands[band][0]) & (f < bands[band][1]))[0]
        if self.exclude==True:
            stft_block_filtered = np.delete(stft_block, indices, axis=0)
            baseline_data_filtered = np.delete(baseline_data, indices, axis=0)
        else:
            stft_block_filtered = stft_block[indices, :]
            baseline_data_filtered = baseline_data[indices, :]
        # stft_block[indices, :] = 0
        
        # shuffle specific band
        # data_to_shuffle = stft_block[indices, :]
        # np.random.shuffle(data_to_shuffle)
        # stft_block[indices, :] = data_to_shuffle
        # print(indices)
        return stft_block_filtered, baseline_data_filtered
    

# 任意组合多个band
class CombineBandDataset(Data.Dataset):
    def __init__(self, HS, path_elec, freq, elec, num_samples, band='None', avg=None):
        super().__init__()
        self.data = [] # 存Ecog数据，shape为（n_samples x n_features)
        self.labels = [] # 存label, shape=（n_samples），cue标记为0，read标记为1
        
        if isinstance(num_samples,int):
            num_samples=[i for i in range(int(num_samples / 5))] # 除4是因为HS69每个电极分6个块的数据 有30的npy文件是因为cue/read/basecue/baseread/baselabel各6个块
        # print(num_samples)
        X_cue,X_read=[],[]
        y_cue,y_read=[],[]
        x_base_cue, x_base_read = [], []
        # print(num_samples)
        for num in num_samples: # num为块的个数

            cue_path = os.path.join(path_elec, f'{num}_data_block_cue.npy')
            read_path = os.path.join(path_elec, f'{num}_data_block_read.npy')
            baseline_cue_path = os.path.join(path_elec, f'{num}_baseline_block_cue.npy')
            baseline_read_path = os.path.join(path_elec, f'{num}_baseline_block_read.npy')

            if os.path.exists(cue_path) and os.path.exists(read_path):
                elec_cue = np.load(cue_path) # (n_task, n_freq, n_timePoint) (60, 501, 375)
                elec_read = np.load(read_path)[:,:,:elec_cue.shape[2]]
                baseline_cue = np.load(baseline_cue_path) 
                baseline_read = np.load(baseline_read_path)
                len_cue = len(elec_cue)
                len_read = len(elec_read)

                if band != 'None':
                    exclu_cue = []
                    exclu_read = []
                    base_cue = []
                    base_read = []
                    for i in range(len_cue):
                        filter_cue, filter_base_cue = self.excluding(elec_cue[i], baseline_cue[i], freq, band)
                        exclu_cue.append(filter_cue)
                        base_cue.append(filter_base_cue)
                    for j in range(len_read):
                        filter_read, filter_base_read = self.excluding(elec_read[j], baseline_read[j], freq, band)
                        exclu_read.append(filter_read)
                        base_read.append(filter_base_read)
                        # exclu_read.append(self.excluding(elec_read[i], baseline_read, freq, band))
                    X_cue.append(exclu_cue)
                    X_read.append(exclu_read)
                    x_base_cue.append(base_cue[:, :, :100]) # 每个block取每个看的onset的前0.2s的数据 拼接在一起作为baseline
                    # x_base_read.append(base_read[:, :, :100])
                else:
                    X_cue.append(elec_cue)
                    X_read.append(elec_read)
                    x_base_cue.append(baseline_cue[:, :, :100])
                    # x_base_read.append(baseline_read[:, :, :100])
                    # x_base_cue = self.expanded_data(baseline_cue, len_cue)
                    # x_base_read = self.expanded_data(baseline_read, len_read)
                    # print(x_base_cue.shape)
        
        
        X_cue=np.abs(np.vstack(X_cue))
        X_read=np.abs(np.vstack(X_read))
        X_base_cue = np.abs(np.vstack(x_base_cue))
        # X_base_read = np.abs(np.vstack(x_base_read))

        X_cue_norm = (X_cue - X_base_cue) / X_base_cue
        # X_read_norm = (X_read - X_base_read) / X_base_read
        X_read_norm = X_cue_norm

        # X_cue_norm = self.z_score_standardize(X_cue, X_base_cue)
        # X_read_norm = self.z_score_standardize(X_read, X_base_read)

        # Compute the mean across the frequency dimension (axis=1)
        if avg == 'avgFreq':
            X_cue_mean = np.mean(X_cue_norm, axis=1)
            X_read_mean = np.mean(X_read_norm, axis=1)
            X=np.concatenate((X_cue_mean, X_read_mean),axis=0)
        # print(f'X_cue_mean.shape: {X_cue_mean.shape}') # (720, 375)

        # Compute the mean across the timesteps dimension (axis=1)
        elif avg == 'avgTime':
            X_cue_mean = np.mean(X_cue_norm, axis=2)
            X_read_mean = np.mean(X_read_norm, axis=2)
            X=np.concatenate((X_cue_mean, X_read_mean),axis=0)

        elif avg == 'norm':
            X=np.concatenate((X_cue_norm, X_read_norm),axis=0)
            X = X.reshape(-1, X.shape[1]*X.shape[2])

        elif avg == 'z-score':
            X_cue_norm = self.z_score_standardize(X_cue, X_base_cue)
            X_read_norm = self.z_score_standardize(X_read, X_base_cue)
            X=np.concatenate((X_cue_norm, X_read_norm),axis=0)
            X = X.reshape(-1, X.shape[1]*X.shape[2])
            
        else:
            X=np.concatenate((X_cue, X_read),axis=0)
            X = X.reshape(-1, X.shape[1]*X.shape[2])

        y_cue=np.array([0]*X_cue.shape[0])
        y_read=np.array([1]*X_read.shape[0])
        y=np.concatenate((y_cue,y_read),axis=0)
        # print(f'x.shape: {X.shape}') # (720, 501, 375)

        self.data = X
        self.labels = y
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a tuple of (data, label) and convert data to a PyTorch tensor
        # print(self.data.shape)
        return torch.tensor(abs(self.data[idx]), dtype=torch.float), torch.tensor(self.labels[idx])
    
    def get_data_labels(self):
        return self.data, self.labels
    
    def z_score_standardize(self, X, X_base):
        """计算 z-score 标准化"""
        mean_X = np.mean(X_base, axis=0)  # 计算基线数据的均值
        std_X = np.std(X_base, axis=0)    # 计算基线数据的标准差
        return (X - mean_X) / std_X
    
    def expanded_data(self, data, expand_size):
        data_0 = data[0]
        expanded_data_0 = np.expand_dims(data_0, axis=0)
        expanded_data = np.tile(expanded_data_0, (expand_size, 1, 1))
        return expanded_data
    
    def excluding(self,stft_block, baseline_data, freq, bands_to_extract):
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

        for band in bands_to_extract:
            if band not in bands:
                raise ValueError(f"Band '{band}' not found in predefined bands.")

        # Combine the indices of all bands to extract
        indices_to_include = np.array([], dtype=int)
        for band in bands_to_extract:
            band_range = bands[band]
            indices = np.where((f >= band_range[0]) & (f < band_range[1]))[0]
            indices_to_include = np.union1d(indices_to_include, indices)

        # print(indices_to_include)
        # Filter the data to include only the selected bands
        stft_block_filtered = stft_block[indices_to_include, :]
        baseline_data_filtered = baseline_data[indices_to_include, :]
        
        return stft_block_filtered, baseline_data_filtered