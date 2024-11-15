#!/Users/DELL/anaconda3/envs/ECOG/python.exe
# -*- coding: utf-8 -*-
# @Time : 2023/9/23 16:06
# @Author : Zhenjie Wang
import matplotlib.pyplot as plt
from scipy import stats
import scipy.io as scio
import numpy as np
import os

EcoG_title = "ECoG_"

onset_time_list = {"HS44": 0, "HS45": 0, "HS47": 0, "HS48": 0, "HS50": 0, "HS54": 0, "HS71": 0, "HS73": 0, "HS76": 0}


class HS_data():

    def __init__(self, HS):
        self.HS = HS
        self.sound_list = self.__get_sound_list()
        self.task_name_list = self.__get_task_name_list()
        self.save_path = "F_values\\" + "HS" + str(self.HS) + "\\"
        self.save_sig_name = "HS" + str(self.HS) + "_F_sig.npy"
        self.save_name = "HS" + str(self.HS) + "_F_values.npy"
        self.__mkdir()
        self.__load_data()
        self.onset_time = int(np.floor(100 * onset_time_list["HS" + str(self.HS)])) + 150
        self.channel = 128 if self.HS == 44 else 256
        self.f_values_dict = {}
        self.sig_elecs_dis = {}

    def __load_data(self):
        mat_path = "HSblockdata\\HS" + str(self.HS) + "_Block_overt_covert.mat"
        self.HSblock = scio.loadmat(mat_path)
        self.HSblock = self.HSblock["Alldata"][0][0]

    def __mkdir(self):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    def __get_sound_list(self):

        if self.HS < 70:
            sound_list = ["ba", "bu", "da", "du", "ga", "gu"]
        else:
            sound_list = ["ba", "da", "ga", "pa", "ta", "ka", "sa", "sha"]

        return sound_list

    def __get_task_name_list(self):

        if self.HS < 70:
            task_name_list = ["overt", "covert"]
        else:
            task_name_list = ["overt", "covert", "cue"]

        return task_name_list

    def calculate_save_f_values(self):
        for task_name in self.task_name_list:
            self.f_values_dict[task_name] = []
            for elec in range(self.channel):
                if self.HS < 70:
                    forward = int(25)
                    backward = int(85)
                    a, b, c, d, e, f = (self.HSblock[EcoG_title + task_name + "_" + sound][:, elec, :].T for sound in
                                        self.sound_list)
                    df = stats.f_oneway(
                        a[int(self.onset_time - forward):int(self.onset_time + backward), :][:,
                        ~np.isnan(a).any(axis=0)].T,
                        b[int(self.onset_time - forward):int(self.onset_time + backward), :][:,
                        ~np.isnan(b).any(axis=0)].T,
                        c[int(self.onset_time - forward):int(self.onset_time + backward), :][:,
                        ~np.isnan(c).any(axis=0)].T,
                        d[int(self.onset_time - forward):int(self.onset_time + backward), :][:,
                        ~np.isnan(d).any(axis=0)].T,
                        e[int(self.onset_time - forward):int(self.onset_time + backward), :][:,
                        ~np.isnan(e).any(axis=0)].T,
                        f[int(self.onset_time - forward):int(self.onset_time + backward), :][:,
                        ~np.isnan(f).any(axis=0)].T,
                    )
                else:
                    forward = int(50)
                    backward = int(150)
                    a, b, c, d, e, f, g, h = (self.HSblock[EcoG_title + task_name + "_" + sound][:, elec, :].T for sound
                                              in self.sound_list)
                    df = stats.f_oneway(
                        a[int(self.onset_time - forward):int(self.onset_time + backward), :][:,
                        ~np.isnan(a).any(axis=0)].T,
                        b[int(self.onset_time - forward):int(self.onset_time + backward), :][:,
                        ~np.isnan(b).any(axis=0)].T,
                        c[int(self.onset_time - forward):int(self.onset_time + backward), :][:,
                        ~np.isnan(c).any(axis=0)].T,
                        d[int(self.onset_time - forward):int(self.onset_time + backward), :][:,
                        ~np.isnan(d).any(axis=0)].T,
                        e[int(self.onset_time - forward):int(self.onset_time + backward), :][:,
                        ~np.isnan(e).any(axis=0)].T,
                        f[int(self.onset_time - forward):int(self.onset_time + backward), :][:,
                        ~np.isnan(f).any(axis=0)].T,
                        g[int(self.onset_time - forward):int(self.onset_time + backward), :][:,
                        ~np.isnan(g).any(axis=0)].T,
                        h[int(self.onset_time - forward):int(self.onset_time + backward), :][:,
                        ~np.isnan(h).any(axis=0)].T)
                self.f_values_dict[task_name].append([df.statistic, df.pvalue])
            self.f_values_dict[task_name] = np.array(self.f_values_dict[task_name])
            print(self.f_values_dict[task_name].shape)

        np.save(self.save_path + self.save_name, self.f_values_dict)

        return self.f_values_dict

    def find_continus(self, aa):
        l1 = []
        total = []
        for x in sorted(set(aa)):
            l1.append(x)
            if x + 1 not in aa:
                total.append(l1)
                l1 = []
        return total

    def get_max_len(self, aa):
        max_len = 0
        for a in aa:
            if len(a) > max_len:
                max_len = len(a)
        return max_len

    def get_sig_elecs(self, alpha=0.01):
        self.sig_elecs_dis = {}

        for i in range(len(self.task_name_list)):
            self.sig_elecs_dis[self.task_name_list[i]] = []
            for elec in range(self.channel):
                seg_index = np.where(self.f_values_dict[self.task_name_list[i]][elec, 1, :] < (alpha))[0]
                if len(seg_index) > 0:
                    max_len = self.get_max_len(self.find_continus(seg_index))
                    if max_len >= 10:
                        self.sig_elecs_dis[self.task_name_list[i]].append(elec)
            print(self.task_name_list[i] + ': ', len(self.sig_elecs_dis[self.task_name_list[i]]))
        np.save(self.save_path + self.save_sig_name, self.sig_elecs_dis)
