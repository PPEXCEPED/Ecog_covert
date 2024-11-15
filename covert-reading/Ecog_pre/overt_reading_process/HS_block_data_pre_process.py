
# -*- coding: utf-8 -*-
# @Time : 2023/11/7 19:53
# @Author : Zhenjie Wang

import os
from copy import deepcopy
from scipy import stats
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import torch
from scipy.stats import f_oneway

channel_num = 256
num_channels = 256
ecog_freq = 100
forward = int(1 * 400)
backward = int(4.5 * 400)
forward_sound = int(1 * 3052)
backward_sound = int(4.5 * 3052)
step = int(4)
length = (forward + backward) // step
time_duration = length / 100
print(length)


def nansem(a, axis=1):
    return np.nanstd(a, axis=1) / np.sqrt(a.shape[axis])


def plot_filled_sem(a, xvals, ax=None, color=None, ylim=None):
    if ax is None:
        fig, ax = plt.subplots()
    mean = np.nanmean(a, axis=1)
    sem = nansem(a, axis=1)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.axhline(0, color='gray', linewidth=0.5)

    if color is not None:
        h = ax.fill_between(xvals, mean - sem, mean + sem, alpha=0.6, color=color)
    else:
        h = ax.fill_between(xvals, mean - sem, mean + sem, alpha=0.6, )

    if ylim is not None:
        ax.set(ylim=ylim)
    ax.set(xlim=(xvals[0], xvals[-1]))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax


def plot_filled_sem_2(a, xvals, ax=None, color=None, ylim=None):  # 仅保留均值线
    if ax is None:
        fig, ax = plt.subplots()
    mean = np.nanmean(a, axis=1)
    sem = nansem(a, axis=1)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.axhline(0, color='gray', linewidth=0.5)

    if color is not None:
        h = ax.fill_between(xvals, mean, mean, alpha=0.6, color=color)
    else:
        h = ax.fill_between(xvals, mean, mean, alpha=0.6, )

    if ylim is not None:
        ax.set(ylim=ylim)
    ax.set(xlim=(xvals[0], xvals[-1]))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax


# check_list = ['ECoG_overt_ba','ECoG_overt_da','ECoG_overt_ga']
# color_list = ['r','g','b']
def ERP_figure(ecogReading, check_list, color_list, mean_erp=False, listen=False):
    sig_test = []

    threshold = 0.05  # 统计检验阈值

    fig, axs = plt.subplots(int(channel_num / 8), 8, figsize=(25, int(channel_num * 25 / 128)))
    # xvals = np.linspace(-0.5, 0.6, 110)
    xvals = np.linspace(-1, 4.5, int(ecog_freq * time_duration))  # 100为采样频率
    if listen:
        xvals = np.linspace(0, 2, int(ecog_freq * 2))  # 100为采样频率

    pidx = channel_num - 1

    for i in range(int(channel_num / 8)):
        for j in range(8):
            sig_test = []
            for k in range(len(check_list)):
                a = ecogReading[check_list[k]][pidx]
                if mean_erp:
                    plot_filled_sem_2(a, xvals, ax=axs[i, j], color=color_list[k])
                else:
                    plot_filled_sem(a, xvals, ax=axs[i, j], color=color_list[k])
                a = a[:, ~np.isnan(a).any(axis=0)].T
                sig_test.append(a)
                if listen:
                    axs[i, j].set_ylim(-2, 5)
                else:
                    axs[i, j].set_ylim(-1, 3)

            #         df=pd.DataFrame(data={'xval':xvals,'oneway':(f_oneway(sig_test[0]).pvalue)})
            #         df_sig = df[df['oneway']<=threshold]
            #             df_non = df[df['oneway']>threshold]
            ax = axs[i, j].twinx()
            #         ax=sns.scatterplot(x = df_sig['xval'],y = -0.3,lw = 2,color = 'black')
            ax.set_yticks([])
            #         ax.set_ylim(-0.4,3.5)
            if listen:
                ax.set_ylim(-2, 5)
            else:
                ax.set_ylim(-1, 3)
            ax.text(0.55, 0.85, str(pidx + 1), transform=ax.transAxes)
            pidx -= 1

    axs = axs.flatten()
    for i, ax in enumerate(axs):
        if listen:
            ax.set(yticklabels=[], xticklabels=[], xticks=[0, 1, 2])

        else:
            ax.set(yticklabels=[], xticklabels=[], xticks=[0, 2, 3.5])
        ax.grid(True)
    # axs[-16].set(yticks=[-1, 0, 1], yticklabels=[-1, 0, 1], xticklabels=[0, 0.5], xlabel="Time (s)", ylabel="High-gamma \n(z-score across block)")

    # plt.show()


def single_elecs_ERP_3pairs(ecogReading, check_list, color_list, pidx):
    threshold = 0.05 / 256
    #     fig, ax = plt.subplots(dpi=300)
    fig, ax = plt.subplots()

    xvals = np.linspace(-1.5, 1.5, int(ecog_freq * time_duration))  # 100为采样频率

    ax.set(xlabel="Time (s)", ylabel="High-gamma \n(z-score)")

    a = ecogReading[check_list[0]][pidx]
    b = ecogReading[check_list[1]][pidx]
    c = ecogReading[check_list[2]][pidx]
    plot_filled_sem(a, xvals, ax=ax, ylim=(-1, 3), color=color_list[0])
    plot_filled_sem(b, xvals, ax=ax, ylim=(-1, 3), color=color_list[1])
    plot_filled_sem(c, xvals, ax=ax, ylim=(-1, 3), color=color_list[2])

    df = pd.DataFrame(data={'xval': xvals, 'oneway': (f_oneway(
        a[:, ~np.isnan(a).any(axis=0)].T,
        b[:, ~np.isnan(b).any(axis=0)].T,
        c[:, ~np.isnan(c).any(axis=0)].T).pvalue)})
    df_sig = df[df['oneway'] <= threshold]
    #         df_non = df[df['oneway']>threshold]

    ax = ax.twinx()
    ax = sns.scatterplot(x=df_sig['xval'], y=-0.3, lw=2, color='black')
    ax.set_yticks([])
    # ax.set_ylim(-0.4,3.5)
    ax.set_ylim(-1, 3)

    # ax.text(0.55, 0.85, 'HS31 channel '+str(pidx+1), transform=ax.transAxes)
    plt.title(str(pidx + 1))
    plt.text(x=0.7, y=2.8, s=check_list[0], fontsize=10, color=color_list[0])
    plt.text(x=0.7, y=2.6, s=check_list[1], fontsize=10, color=color_list[1])
    plt.text(x=0.7, y=2.4, s=check_list[2], fontsize=10, color=color_list[2])


def single_elecs_ERP_2pairs(ecogReading, check_list, color_list, pidx):
    threshold = 0.05 / 256
    #     fig, ax = plt.subplots(dpi=300)
    fig, ax = plt.subplots()
    xvals = np.linspace(-1.5, 1.5, int(ecog_freq * time_duration))  # 100为采样频率

    ax.set(xlabel="Time (s)", ylabel="High-gamma \n(z-score)")

    b = ecogReading[check_list[0]][pidx]
    c = ecogReading[check_list[1]][pidx]
    plot_filled_sem(b, xvals, ax=ax, ylim=(-1, 3), color=color_list[0])
    plot_filled_sem(c, xvals, ax=ax, ylim=(-1, 3), color=color_list[1])

    df = pd.DataFrame(data={'xval': xvals, 'oneway': (stats.ttest_ind(  # 两独立样本t检验
        b[:, ~np.isnan(b).any(axis=0)].T,
        c[:, ~np.isnan(c).any(axis=0)].T).pvalue)})
    df_sig = df[df['oneway'] <= threshold]
    #         df_non = df[df['oneway']>threshold]

    ax = ax.twinx()
    ax = sns.scatterplot(x=df_sig['xval'], y=-0.3, lw=2, color='black')
    ax.set_yticks([])
    # ax.set_ylim(-0.4,3.5)
    ax.set_ylim(-1, 3)

    # ax.text(0.55, 0.85, 'HS31 channel '+str(pidx+1), transform=ax.transAxes)
    plt.title(str(pidx + 1))
    plt.text(x=0.7, y=2.8, s=check_list[0], fontsize=10, color=color_list[0])
    plt.text(x=0.7, y=2.6, s=check_list[1], fontsize=10, color=color_list[1])


def single_elecs_ERP_all_pairs(ecogReading, check_list, color_list, pidx):
    #     threshold = 0.05/256
    #     fig, ax = plt.subplots(dpi=300)
    fig, ax = plt.subplots()
    xvals = np.linspace(-1.5, 1.5, int(ecog_freq * time_duration))  # 100为采样频率

    ax.set(xlabel="Time (s)", ylabel="High-gamma \n(z-score)")

    for i in range(len(check_list)):
        a = ecogReading[check_list[i]][pidx]
        plot_filled_sem(a, xvals, ax=ax, ylim=(-1, 3), color=color_list[i])
        plt.text(x=0.7, y=2.8 - 0.2 * i, s=check_list[i], fontsize=10, color=color_list[i])

    #     df=pd.DataFrame(data={'xval':xvals,'oneway':(f_oneway(
    #         a[:, ~np.isnan(a).any(axis=0)].T,
    #         b[:, ~np.isnan(b).any(axis=0)].T,
    #         c[:, ~np.isnan(c).any(axis=0)].T).pvalue)})
    #     df_sig = df[df['oneway']<=threshold]
    #     #         df_non = df[df['oneway']>threshold]

    ax = ax.twinx()
    #     ax=sns.scatterplot(x = df_sig['xval'],y = -0.3,lw = 2,color = 'black')
    ax.set_yticks([])
    # ax.set_ylim(-0.4,3.5)
    ax.set_ylim(-1, 3)

    # ax.text(0.55, 0.85, 'HS31 channel '+str(pidx+1), transform=ax.transAxes)
    plt.title(str(pidx + 1))


def single_elecs_mean_ERP_all_pairs(ecogReading, check_list, color_list, pidx):
    #     threshold = 0.05/256
    #     fig, ax = plt.subplots(dpi=300)
    fig, ax = plt.subplots()
    xvals = np.linspace(-1.5, 1.5, int(ecog_freq * time_duration))  # 100为采样频率

    ax.set(xlabel="Time (s)", ylabel="High-gamma \n(z-score)")

    for i in range(len(check_list)):
        a = ecogReading[check_list[i]][pidx]
        plot_filled_sem_2(a, xvals, ax=ax, ylim=(-1, 3), color=color_list[i])
        plt.text(x=0.7, y=2.8 - 0.2 * i, s=check_list[i], fontsize=10, color=color_list[i])

    #     df=pd.DataFrame(data={'xval':xvals,'oneway':(f_oneway(
    #         a[:, ~np.isnan(a).any(axis=0)].T,
    #         b[:, ~np.isnan(b).any(axis=0)].T,
    #         c[:, ~np.isnan(c).any(axis=0)].T).pvalue)})
    #     df_sig = df[df['oneway']<=threshold]
    #     #         df_non = df[df['oneway']>threshold]

    ax = ax.twinx()
    #     ax=sns.scatterplot(x = df_sig['xval'],y = -0.3,lw = 2,color = 'black')
    ax.set_yticks([])
    # ax.set_ylim(-0.4,3.5)
    ax.set_ylim(-1, 3)

    # ax.text(0.55, 0.85, 'HS31 channel '+str(pidx+1), transform=ax.transAxes)
    plt.title(str(pidx + 1))


class HSblock_pre_process:
    def __init__(self, HS, workspace_path):
        self.HS = HS
        self.fs_ecog = 400
        self.fs_sound = 24414
        self.HS_path = os.path.join(workspace_path, "HS_" + str(HS))
        dir_name, file_name = os.path.split(workspace_path)

        self.block_path = os.path.join(dir_name, 'blocked_data')
        self.sound_path = os.path.join(self.HS_path, "ECoG")
        self.mat_path = os.path.join(self.HS_path, "ECoG")
        self.marker_path = os.path.join(self.HS_path, "marker")
        self.sound_syn_list = glob.glob(self.sound_path + "/*syn.mat")
        self.sound_list = glob.glob(self.sound_path + "/*sound.mat")
        self.mat_path_list = glob.glob(self.mat_path + "/*150.mat")
        self.onset_name = ["1", "4", "5", "6", "7", "8", "12", "11", "10", "9"]
        self.blockmat = {}
        self.blockmat_sound = {}
        self.blockmat_sound_syn = {}
        self.delay_list = {}
        self.block_name, self.onset_cue_time, self.onset_read_time, self.start_onset_time = {}, {}, {}, {}
        self.syn_point = {}
        self.syn_point['HS62'] = {"0": [1, 2, 3, 4, 5], "1": [1, 2, 3, 4, 5],
                                  "2": [1, 2, 3, 4, 5], "3": [1, 2, 3, 4],
                                  "4": [0, 1, 2, 3], "5": [0],
                                  "6": [0, 1, 2, 3, 4, 5]}
        self.syn_point['HS68'] = {"0": [1, 2, 3, 4, 5], "1": [1, 2, 3, 4, 5],
                                  "2": [1, 2, 3, 4, 5], "3": [1, 2, 3, 4, 5],
                                  "4": [1, 2, 3, 4, 5], "5": [1, 2, 3, 4, 5], }
        self.syn_point['HS69'] = {"0": [1, 2, 3, 4, 5], "1": [1, 2, 3, 4, 5],
                                  "2": [1, 2, 3, 4, 5], "3": [1, 2, 3, 4, 5],
                                  "4": [1, 2, 3, 4, 5], "5": [1, 2, 3, 4, 5], }
        self.syn_point['HS75'] = {"0": [1, 2, 3, 4, 5], "1": [1, 2, 3, 4, 5],
                                  "2": [1, 2, 3, 4, 5], "3": [1, 2, 3, 4, 5],
                                  "4": [1, 2, 3, 4, 5], "5": [1, 2, 3, 4, 5], }
        self.syn_point['HS79'] = {"0": [0], "1": [1, 2, 3, 4, 5],
                                  "2": [1, 2, 3, 4, 5], "3": [1, 2, 3, 4, 5],
                                  "4": [1, 2, 3, 4, 5], "5": [0, 1, 2]}
        self.syn_point['HS82'] = {"0": [1, 2, 3, 4, 5], "1": [1, 2, 3, 4, 5],
                                  "2": [1, 2, 3, 4, 5], "3": [1, 2, 3, 4, 5],
                                  "4": [1, 2, 3, 4, 5], "5": [1, 2, 3, 4, 5], }
        self.syn_point['HS83'] = {"0": [1, 2, 3, 4, 5], "1": [1, 2, 3, 4, 5],
                                  "2": [1, 2, 3, 4, 5], "3": [1, 2, 3, 4, 5],
                                  "4": [1, 2, 3, 4, 5], "5": [1, 2, 3, 4, 5], }
        self.syn_point['HS84'] = {"0": [1, 2, 3, 4, 5], "1": [1, 2, 3, 4, 5],
                                  "2": [1, 2, 3, 4, 5], "3": [1, 2, 3, 4, 5],
                                  "4": [1, 2, 3, 4, 5], "5": [1, 2, 3, 4, 5], }
        self.syn_point['HS85'] = {"0": [1, 2, 3, 4, 5], "1": [1, 2, 3, 4, 5],
                                  "2": [1, 2, 3, 4, 5], "3": [1, 2, 3, 4, 5],
                                  "4": [1, 2, 3, 4, 5], "5": [1, 2, 3, 4, 5], }
        self.syn_point['HS86'] = {"0": [1, 2, 3, 4, 5], "1": [1, 2, 3, 4, 5],
                                  "2": [1, 2, 3, 4, 5], "3": [1, 2, 3, 4, 5],
                                  "4": [1, 2, 3, 4, 5], "5": [1, 2, 3, 4, 5], }
        self.block_list = range(len(self.mat_path_list))
        self.block_num = len(self.mat_path_list)
        self.average = np.zeros([self.block_num, 256])
        self.std = np.zeros([self.block_num, 256])
        self.average_listen = np.zeros([self.block_num, 256])
        self.std_listen = np.zeros([self.block_num, 256])
        self.task_keys = ['功课', '力果', '宫客', '作业', 'gōng kè', '树叶', '对十', '数页', '绿草', 'shù yè']

        self.sound_listen_list = ["功课", "力果", "作业", "树叶", "对十", "绿草"]
        self.list_num = [[1, 4, 2, 3, 6, 5],
                         [2, 5, 3, 4, 1, 6],
                         [1, 3, 2, 6, 5, 4],
                         [6, 5, 1, 4, 3, 2],
                         [3, 6, 5, 2, 4, 1],
                         [3, 4, 5, 1, 6, 2],
                         ]
        self.list_num = torch.tensor(self.list_num)

        # 针对各个特殊情况的处理

    def __load_block_mat(self):
        for i in range(len(self.mat_path_list)):
            self.blockmat[str(i)] = scio.loadmat(os.path.join(self.mat_path, self.mat_path_list[i]))

        for i in range(len(self.mat_path_list)):
            self.blockmat_sound_syn[str(i)] = scio.loadmat(self.sound_syn_list[i])
            self.blockmat_sound[str(i)] = scio.loadmat(self.sound_list[i])

        # if self.HS == 79:
        #     self.blockmat[str(5)] = self.blockmat[str(4)]
        #     self.blockmat_sound_syn[str(5)] = deepcopy(self.blockmat_sound_syn[str(4)])
        #     self.blockmat_sound[str(5)] = self.blockmat_sound[str(4)]
        #     self.blockmat_sound_syn[str(5)]['data'][0, :8000000] = 0
        # if self.HS == 82:
        #
        #     self.blockmat_sound_syn[str(2)]['data'][0, :1100000] = 0

    def __get_syn_point(self, block, thresholding=0.17):

        peakpos = np.where(self.blockmat_sound_syn[str(block)]['data'].T > thresholding)[0]
        localpos = []
        fs = 24414
        space = 1e6
        for i in range(len(peakpos)):
            if i == 0:
                localpos.append(peakpos[i])
            if i > 0:
                if peakpos[i] - peakpos[i - 1] > space:
                    localpos.append(peakpos[i])
        return torch.tensor(localpos)

    def plot_block_sound_data(self):
        self.__load_block_mat()
        # plt.figure(figsize=(15, 15))
        for i in range(len(self.blockmat)):
            # plt.subplot(int(np.ceil(len(self.blockmat) / 2)), 2, i + 1)
            block = i
            localpos = self.__get_syn_point(block, 0.01)

            # plt.plot(self.blockmat_sound[str(block)]['data'], alpha=0.5)
            # plt.plot(self.blockmat_sound_syn[str(block)]['data'].T, alpha=0.5)
            # plt.title("session: " + str(block))
            # plt.scatter(localpos, 0.17 * np.ones([len(localpos)]), c='r')

    def corrected_eprime_ecog(self):
        # plt.figure(figsize=(10, 6))
        figx = 1

        fs = 24414
        for block in range(len(self.blockmat)):
            # plt.subplot(int(np.ceil(len(self.blockmat) / 2)), 2, figx)
            figx += 1
            if self.HS == 79 and block == 0:
                _, _, _, start_onset = self.__get_true_onset(block, True)
                localpos = self.__get_syn_point(block)
            else:
                _, _, _, start_onset = self.__get_true_onset(block)
                localpos = self.__get_syn_point(block)
            # plt.scatter(start_onset / 1000, torch.ones_like(start_onset))
# 
            # plt.scatter(localpos / fs, 2 * torch.ones_like(localpos))
            # plt.scatter(localpos / fs + self.__get_delay(start_onset, localpos, block) / fs,
                        # 1.5 * torch.ones_like(localpos))
            ecog_c = localpos / fs + self.__get_delay(start_onset, localpos, block) / fs
            ecog = localpos / fs

            # plt.legend(["eprime", "ecog", "ecog_corrected"], loc="lower left")

            # for i in range(1, len(2 * torch.ones_like(localpos))):
            #     plt.arrow(ecog[i], 2, ecog_c[i] - ecog[i], -0.5, color='k', head_width=0.2, lw=0.5,
            #               length_includes_head=True)

            self.delay_list[str(block)] = self.__get_delay(start_onset, localpos, block)

    def plot_correct_wav(self):
        # plt.figure(figsize=(25, 10))
        #

        figx = 1
        for block in range(len(self.blockmat)):
            # plt.subplot(int(np.ceil(len(self.blockmat) / 2)), 2, figx)
            figx += 1
            if self.HS == 79 and block == 0:
                self.block_name[str(block)], self.onset_cue_time[str(block)], self.onset_read_time[str(block)], \
                    self.start_onset_time[
                        str(block)] = self.__get_true_onset(block, cue=True)
            else:
                self.block_name[str(block)], self.onset_cue_time[str(block)], self.onset_read_time[str(block)], \
                    self.start_onset_time[
                        str(block)] = self.__get_true_onset(block)
            localpos = self.__get_syn_point(block)

            # plt.plot(self.blockmat_sound[str(block)]['data'], alpha=0.5)
            # plt.scatter(self.start_onset_time[str(block)] * 24.414 - self.delay_list[str(block)],
            #             torch.ones_like(self.start_onset_time[str(block)]), c='r')
            # print(np.array([on1,on2,on3])*24.414+delta)

    def zscore_init(self):
        # 针对只有听的block进行不同的初始化，没有听则从第一个点开始初始化
        for i in range(len(self.blockmat)):
            if self.syn_point[f"HS{self.HS}"][str(i)][0] == 0 and len(self.syn_point[f"HS{self.HS}"][str(i)]) > 1:
                listen_end_time = 0
            else:
                listen_end_time = 80 * self.fs_ecog
            for j in range(256):
                self.average[i, j] = np.average(self.blockmat[str(i)]['bands'][j][listen_end_time:])
                self.std[i, j] = np.std(self.blockmat[str(i)]['bands'][j][listen_end_time:])

        # 如果没有听那么则不需要进行听的初始化
        for i in range(len(self.blockmat)):
            if self.syn_point[f"HS{self.HS}"][str(i)][0] == 0 and len(self.syn_point[f"HS{self.HS}"][str(i)]) > 1:

                continue
            else:
                listen_end_time = 80 * self.fs_ecog
                for j in range(256):
                    self.average_listen[i, j] = np.average(self.blockmat[str(i)]['bands'][j][:listen_end_time])
                    self.std_listen[i, j] = np.std(self.blockmat[str(i)]['bands'][j][:listen_end_time])

    def extract_ecog2HSblock(self):

        self.__align_block_cue_read_len()
        forward = int(1 * self.fs_ecog)
        backward = int(4.5 * self.fs_ecog)

        forward_sound = int(1 * self.fs_sound)
        backward_sound = int(4.5 * self.fs_sound)
        step = int(4)
        length = (forward + backward) // step
        n_trials = 36
        if self.HS == 79:
            n_trials = 32
        print(length)
        # self.keys = [' 果汁', ' 商店', ' 粮食', ' 淉汥',' 啇扂', ' 悢喰', ' 裹知',' 伤电',' 量时',' guozhi', ' shangdian', ' liangshi',' 饮料',' 超市', ' 稻米', ' ワテ', ' ヅヂ\xa0', ' ギグ',  ' 滞峦',  ' 琼殷', ' 梧漾']
        # self.keys_two = [' 上', ' 左', ' 下', ' 右']
        # self.keys_all = [' 果汁', ' 商店', ' 粮食', ' 淉汥',' 啇扂', ' 悢喰', ' 裹知',' 伤电',' 量时',' guozhi', ' shangdian', ' liangshi',' 饮料',' 超市', ' 稻米', ' ワテ', ' ヅヂ\xa0', ' ギグ',  ' 滞峦',  ' 琼殷', ' 梧漾',' 上', ' 左', ' 下', ' 右']
        # self.keys = ['课计', 'guǒ zhī', '裹知', '火车', '高铁', '果汁', '했밞', '糕帖', 'gāo tiě', '亮钟', '饮料']
        self.keys = ['功课', '力果', '宫客', '作业', 'gōng kè', '树叶', '对十', '数页', '绿草', 'shù yè']
        # self.keys_all = [i+" reading" for i in self.keys]
        # self.keys_all.extend(self.keys)

        self.hs_block = dict([(k, np.zeros([n_trials, 256, length])) for k in self.keys])
        count_dict = dict([(k, 0) for k in self.keys])

        # 获取第一部分听的ecog数据
        for block in self.block_list:
            for i in self.sound_listen_list:
                self.hs_block[i + " listen"] = []

        for block in self.block_list:
            localpos = self.__get_syn_point(block)
            if self.syn_point[f"HS{self.HS}"][str(block)][0] == 0 and len(
                    self.syn_point[f"HS{self.HS}"][str(block)]) > 1:
                continue
            list_start_point = int(localpos[0] / self.fs_sound * self.fs_ecog)
            time_point = list_start_point
            for i in range(6):
                for j in range(6):
                    task = self.sound_listen_list[int(self.list_num[i, j]) - 1]
                    listen_block = []
                    for k in range(256):
                        listen_silce = self.__zscore_listen(
                            self.blockmat[str(block)]['bands'][k][time_point:time_point + int(2 * self.fs_ecog):4],
                            block, k)
                        listen_block.append(listen_silce)
                    self.hs_block[task + " listen"].append(listen_block)
                    time_point = time_point + int(2 * self.fs_ecog)

        # 获取第二部分看和读的ecog数据
        for block in self.block_list:
            if self.HS == 62 and block > 3:
                break
            for i in range(len(self.keys)):
                self.hs_block[self.block_name[str(block)][i] + " sound"] = []

        for block in self.block_list:
            if self.HS == 62 and block > 3:
                break
            print("block: ", block)
            for i in range(len(self.block_name[str(block)])):
                # print(block_name[str(block)][i],i)
                for j in range(256):
                    self.hs_block[self.block_name[str(block)][i]][count_dict[self.block_name[str(block)][i]]][j][
                    :] = self.__zscore(self.blockmat[str(block)]['bands'][j][
                                       self.__transfer(self.onset_cue_time[str(block)][i],
                                                       block) - forward:self.__transfer(
                                           self.onset_cue_time[str(block)][i], block) + backward:step], block, j)

                sound_slice = self.blockmat_sound[str(block)]['data'][
                              self.__transfer_sound(self.onset_cue_time[str(block)][i],
                                                    block) - forward_sound:self.__transfer_sound(
                                  self.onset_cue_time[str(block)][i],
                                  block) + backward_sound, 0]

                self.hs_block[self.block_name[str(block)][i] + " sound"].append(sound_slice)
                count_dict[self.block_name[str(block)][i]] += 1

        # 将list 转为numpy格式
        for i in self.hs_block:
            self.hs_block[i] = np.array(self.hs_block[i])

        for i in range(len(self.keys)):
            self.hs_block[self.keys[i]] = np.swapaxes(np.swapaxes(self.hs_block[self.keys[i]], 0, 2), 0, 1)
        for i in self.sound_listen_list:
            self.hs_block[i + " listen"] = np.swapaxes(np.swapaxes(self.hs_block[i + " listen"], 0, 2), 0, 1)

        return self.hs_block

    def QC_delete(self,QCmat):

        for i in QCmat:
            dele_slice_list = [6 * ind[0] + ind[1] for ind in QCmat[i]]
            self.hs_block[i] = np.delete( self.hs_block[i], dele_slice_list, axis=2)
            self.hs_block[i + " sound"] = np.delete(self.hs_block[i + " sound"], dele_slice_list, axis=0)

    def save_npy(self):
        np.save(self.block_path + "/HS" + str(self.HS) + "block.npy", self.hs_block)

    def __find_onset(self, x, start_onset, cue=False):
        if cue:
            if "Slidefornone14.OnsetTime:" in x:
                pos = x.find(':')
                start_onset.append(int(x[pos + 2:]) + 977)
        else:
            for i in self.onset_name:
                if f"Slidefornone{i}.OnsetTime:" in x:
                    pos = x.find(':')
                    start_onset.append(x[pos + 2:])
        return start_onset

    def __get_true_onset(self, block, cue=False):
        filedir = os.listdir(self.marker_path)
        textname = self.marker_path + "/sound2group_final-" + str(self.HS) + "-" + str(block) + ".txt"
        block_name = []
        onset_cue_time = []
        onset_read_time = []
        onset_delay = []
        start_onset = []
        flag = 100
        flag2 = 100
        with open(textname, 'r', encoding='utf-16-le') as f:
            for ann in f.readlines():
                ann = ann.strip('\n')  # 去除文本中的换行符
                if "textname" in ann:
                    pos = ann.find(':')
                    block_name.append(ann[pos + 2:])
                    flag = 0
                    flag2 = 0
                # if ("Slidetext" in ann) & ("OnsetDelay"in ann) & (flag <2) :
                #     pos = ann.find(':')
                #     onset_delay.append(ann[pos+2:])
                #     flag += 1
                if ("Slidetext" in ann) & ("OnsetTime" in ann) & (flag < 1):
                    pos = ann.find(':')
                    onset_cue_time.append(ann[pos + 2:])
                    flag += 1
                if ("SlidetextRead" in ann) & ("OnsetTime" in ann) & (flag2 < 1):
                    pos = ann.find(':')
                    onset_read_time.append(ann[pos + 2:])
                    flag2 += 1

                start_onset = self.__find_onset(ann, start_onset, cue)

        # for i in range(len(self.block_name)):
        #     onset_time[i] = int(onset_time[i]) - int(onset_delay[i])

        start_onset = [int(i) for i in start_onset]
        onset_read_time = [int(i) for i in onset_read_time]
        onset_cue_time = [int(i) for i in onset_cue_time]
        # for i in range(0,16,2):
        #     self.start_onset_time.append(start_onset[i+1]-start_onset[i])
        print("oneset_cue_point:", len(onset_cue_time), onset_cue_time)
        print("oneset_read_point:", len(onset_read_time), onset_read_time)
        print("self.block_name:", len(block_name), block_name)
        print("syn_point:", start_onset)
        print(textname)
        return block_name, torch.tensor(onset_cue_time), torch.tensor(onset_read_time), torch.tensor(
            start_onset)

    def __get_delay(self, eprime, ecog, block):
        fs = 24414
        res = eprime.unsqueeze(-1) / 1000 * fs - ecog
        return torch.mean(res.diag()[self.syn_point[f'HS{self.HS}'][str(block)]])

    def __dele_last(self, block, x):
        self.block_name[str(block)] = self.block_name[str(block)][:-x]
        self.onset_cue_time[str(block)] = self.onset_cue_time[str(block)][:-x]
        self.onset_read_time[str(block)] = self.onset_read_time[str(block)][:-x]

    def __transfer(self, x, block):
        return int((int(x) * 24.414 - self.delay_list[str(block)]) / 24414 * 400)

    def __transfer_sound(self, x, block):
        return int((int(x) * 24.414 - self.delay_list[str(block)]))

    def __zscore(self, x, i, j):
        return (x - self.average[i, j]) / self.std[i, j]

    def __zscore_listen(self, x, i, j):
        return (x - self.average_listen[i, j]) / self.std_listen[i, j]

    def __align_block_cue_read_len(self):
        if self.HS == 62:
            self.block_name[str(4)] = self.block_name[str(4)][:-1]

            self.__dele_last(4, 2)

            self.block_name[str(5)] = self.block_name[str(5)][:-1]
            self.onset_cue_time[str(5)] = self.onset_cue_time[str(5)][:-1]

            self.__dele_last(5, 20)
            print(len(self.onset_cue_time[str(5)]))
            print(len(self.block_name[str(4)]))
            print(len(self.block_name[str(5)]))
        if self.HS == 79:
            self.block_name[str(5)] = self.block_name[str(5)][:-1]
            self.onset_cue_time[str(5)] = self.onset_cue_time[str(5)][:-1]

            print(len(self.onset_cue_time[str(5)]))
            print(len(self.block_name[str(4)]))
            print(len(self.block_name[str(5)]))
