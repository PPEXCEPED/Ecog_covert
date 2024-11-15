
import scipy.io as scio
import itertools
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from scipy.stats import pearsonr
import seaborn as sns

################################
#本文件用于计算两个序列间最大correlation，计算pair，绘制onset、peak、offset的图

def cross_correlation_matrix_dot(sp_1, sp_2, lag, normalized):
    total_matrix = np.zeros([1 + 2 * lag, len(sp_1) + 2 * lag])
    total_matrix_sp2 = np.zeros([len(sp_2) + 2 * lag, 1])
    for i in range(1 + 2 * lag):
        total_matrix[i, i:i + len(sp_1)] = sp_1

    total_matrix_sp2[lag:lag + len(sp_2), 0] = sp_2

    result = total_matrix @ total_matrix_sp2
    result = np.squeeze(result) * normalized
    return result
def cross_correlation_matrix_pearsonr(sp_1, sp_2, lag):
    total_matrix = np.zeros([1 + 2 * lag, len(sp_1) + 2 * lag])

    r,p = np.zeros([1 + 2 * lag]), np.zeros([1 + 2 * lag])
    for i in range(1 + 2 * lag):
        total_matrix[i, i:i + len(sp_1)] = sp_1
        if len(total_matrix[i,lag:i + len(sp_1)])<=len(sp_2[0:i+len(sp_2)-lag]):
            r[i],p[i] = pearsonr(total_matrix[i,lag:i + len(sp_1)],sp_2[:i+len(sp_2)-lag])
        else:
            r[i], p[i] = pearsonr(total_matrix[i, i:len(sp_1)+lag], sp_2[i - lag:])


    return r,p
def get_iter():
    sound_list1 = ["ba", "bu", "da", "du", "ga", "gu"]
    task_name_list1 = ["overt", "covert"]
    sound_list2 = ["ba", "da", "ga", "pa", "ta", "ka", "sa", "sha"]
    task_name_list2 = ["overt", "covert", "cue"]
    iter1 = list(itertools.product(task_name_list1, sound_list1))
    iter1 = ['_' + item1[0] + '_' + item1[1] for item1 in iter1]
    iter2 = list(itertools.product(task_name_list2, sound_list2))
    iter2 = ['_' + item2[0] + '_' + item2[1] for item2 in iter2]
    return iter1,iter2

def pairs(lag, HS_list, clean_path_data,correlation = "pearsonr"):
    iter1, iter2 =get_iter()
    onset = 150
    normalized = np.zeros([2 * lag + 1])

    for num in HS_list:
        print("HS" + str(num) + " start: ")
        lg_path = clean_path_data + "/lags/"
        if os.path.exists(lg_path):
            pass
        else:
            os.mkdir(lg_path)

        temp = scio.loadmat(clean_path_data + f"/HSblockdata/HS{num}_Block_overt_covert.mat")["Alldata"][0][0]
        elec_total_r = {}
        elec_total_p = {}

        if num < 70:
            iter = iter1
            forward = int(40)
            backward = int(120)
        else:
            iter = iter2
            forward = int(50)
            backward = int(150)
        if correlation != "pearsonr":
            for i in range(2 * lag + 1):
                normalized[i] = 1 / (forward + backward - abs((lag - i)))

        for term in iter:
            ecog = temp['ECoG' + term]
            elec_num = ecog.shape[1]
            elec_total_r[f'{term}'] = {}
            elec_total_p[f'{term}'] = {}
            for p in range(elec_num):
                for q in range(p+1, elec_num):
                    # trial_list = [cross_correlation_matrix_dot(ecog[j, p, onset - forward:onset + backward],
                    #                                        ecog[j, q, onset - forward:onset + backward], lag,
                    #                                        normalized) for j in
                    #               range(ecog.shape[0])]
                    trial_list_r = []
                    trial_list_p = []

                    trial_list_r, trial_list_p = cross_correlation_matrix_pearsonr(np.mean(ecog[:, p, onset - forward:onset + backward],axis=0),
                                                           np.mean(ecog[:, q, onset - forward:onset + backward],axis=0), lag)
                    elec_total_r[f'{term}'][f'{p}' + '_' + f'{q}'] = np.array(trial_list_r)
                    elec_total_p[f'{term}'][f'{p}' + '_' + f'{q}'] = np.array(trial_list_p)

            print(num, term, " complete")
        np.save(lg_path  + f"/HS{num}_Block_lag_r.npy", elec_total_r)
        np.save(lg_path + f"/HS{num}_Block_lag_p.npy", elec_total_p)

        print(num, " complete")
def pairs_sound_avg(lag, HS_list, clean_path_data,correlation = "pearsonr"):

    onset = 150
    normalized = np.zeros([2 * lag + 1])

    for num in HS_list:
        print("HS" + str(num) + " start: ")
        lg_path = clean_path_data + "/lags/"
        if os.path.exists(lg_path):
            pass
        else:
            os.mkdir(lg_path)

        temp = scio.loadmat(clean_path_data + f"/HSblockdata/HS{num}_Block_overt_covert.mat")["Alldata"][0][0]
        elec_total_r = {}
        elec_total_p = {}

        if num < 70:
            sound_list = ["ba", "bu", "da", "du", "ga", "gu"]
            task_name_list = ["overt", "covert"]

            forward = int(40)
            backward = int(120)
        else:
            sound_list = ["ba", "da", "ga", "pa", "ta", "ka", "sa", "sha"]
            task_name_list = ["overt", "covert", "cue"]
            forward = int(50)
            backward = int(150)
        if correlation != "pearsonr":
            for i in range(2 * lag + 1):
                normalized[i] = 1 / (forward + backward - abs((lag - i)))

        for term in task_name_list:
            for sound_index in range(len(sound_list)):
                if sound_index == 0:
                    ecog = temp['ECoG_' +term+'_'+sound_list[sound_index]]
                else:
                    ecog = np.concatenate((ecog,temp['ECoG_' +term+'_'+sound_list[sound_index]]),axis=0)
            print(ecog.shape)
            elec_num = ecog.shape[1]
            elec_total_r[f'{term}'] = {}
            elec_total_p[f'{term}'] = {}
            for p in range(elec_num):
                for q in range(p+1, elec_num):
                    # trial_list = [cross_correlation_matrix_dot(ecog[j, p, onset - forward:onset + backward],
                    #                                        ecog[j, q, onset - forward:onset + backward], lag,
                    #                                        normalized) for j in
                    #               range(ecog.shape[0])]
                    trial_list_r = []
                    trial_list_p = []

                    trial_list_r, trial_list_p = cross_correlation_matrix_pearsonr(np.mean(ecog[:, p, onset - forward:onset + backward],axis=0),
                                                           np.mean(ecog[:, q, onset - forward:onset + backward],axis=0), lag)
                    elec_total_r[f'{term}'][f'{p}' + '_' + f'{q}'] = np.array(trial_list_r)
                    elec_total_p[f'{term}'][f'{p}' + '_' + f'{q}'] = np.array(trial_list_p)

            print(num, term, " complete")

        # np.save(lg_path  + f"/HS{num}_Block_lag_r_sound_avg.npy", elec_total_r)
        # np.save(lg_path + f"/HS{num}_Block_lag_p_sound_avg.npy", elec_total_p)


        print(num, " complete")


def lag_data_generate(HS_list, clean_data_path):
    place_list = []
    avgECoG = {}
    sound_list1 = ["ba", "bu", "da", "du", "ga", "gu"]
    task_name_list1 = ["overt", "covert"]
    sound_list2 = ["ba", "da", "ga", "pa", "ta", "ka", "sa", "sha"]
    task_name_list2 = ["overt", "covert", "cue"]
    for task in task_name_list2:
        avgECoG[task] = {}

    for HS in HS_list:
        # xycolor = scio.loadmat(clean_data_path + "/elecs/warped/HS" + str(HS) + "_elecs_all_warped.mat")["anatomy"]
        # elec_sig = np.load(clean_data_path+"/elecs/elec_sig/" + str(HS) + "sig_elecs.npy",allow_pickle=True).item()

        xycolor = scio.loadmat(clean_data_path + "/elecs/warped/HS" + str(HS) + "_elecs_all_warped.mat")["anatomy"]
        elec_sig = np.load(clean_data_path + "/elecs/elec_sig/" + str(HS) + "sig_elecs.npy", allow_pickle=True).item()

        place_list.extend(np.unique(xycolor[:, 3]))

        for task in task_name_list2:
            for i in range(len(place_list)):
                if place_list[i][0] not in avgECoG[task].keys():
                    avgECoG[task][place_list[i][0]] = []

        clean_data_path_block = os.path.join(clean_data_path, "HSblockdata")
        file_name = "HS" + str(HS) + "_Block_overt_covert.mat"
        HSblock = scio.loadmat(os.path.join(clean_data_path_block, file_name))
        HSblock = HSblock["Alldata"][0][0]
        n_chans = xycolor.shape[0]
        # 为各个place各个task增加一行对特定音节该电极上的平均
        for i in range(n_chans):
            if HS >54:
                for task in task_name_list2:
                    if i in elec_sig[task]:
                        for sound in sound_list2:
                            avgECoG[task][xycolor[i, 3][0]].append(np.mean(HSblock["ECoG_"+task+"_"+sound][:, i, :], axis=0))
            else:
                for task in task_name_list1:
                    if i in elec_sig[task]:
                        for sound in sound_list1:
                            avgECoG[task][xycolor[i, 3][0]].append(np.mean(HSblock["ECoG_"+task+"_"+sound][:, i, :], axis=0))
    place_list = np.unique(place_list)
    return place_list, avgECoG


def lag_plot(HS_list, clean_data_path,method):
    n_windows = 5
    # 选择是否高斯平滑：如果高斯平滑n为多少
    place_list, avgECoG = lag_data_generate(HS_list, clean_data_path)
    gauss = False
    sound_list2 = ["ba", "da", "ga", "pa", "ta", "ka", "sa", "sha"]
    task_name_list2 = ["overt", "covert", "cue"]
    for task in task_name_list2:
        for i in range(len(place_list)):
            avgECoG[task][place_list[i]] = np.array(avgECoG[task][place_list[i]])
            if avgECoG[task][place_list[i]].shape[0] == 1:
                avgECoG[task][place_list[i]] = np.expand_dims(avgECoG[task][place_list[i]],0)
            if gauss:
                for j in range(avgECoG[task][place_list[i]].shape[0]):
                    avgECoG[task][place_list[i]][j] = np.convolve(avgECoG[task][place_list[i]][j], np.ones([n_windows]) / n_windows, mode="same")


    forward = int(40)
    backward = int(150)
    task_dict = {}

    for task in task_name_list2:
        blank_list = []
        onset = {}
        onset_t = {}
        peak = {}
        peak_t = {}
        offset = {}
        offset_t = {}
        for i in range(len(place_list)):
            temp = avgECoG[task][place_list[i]]
            if len(temp) == 0:
                blank_list.append(i)
                continue
            # temp[temp<0]=0
            onset[place_list[i]] = np.argmax(np.diff(temp, axis=1)[:, -forward+150:backward+150], axis=1)
            num = 100
            onset_t[place_list[i]] = (np.argmax(np.diff(temp, axis=1)[:, -forward + 150:backward + 150],
                                                axis=1) - forward) / num
            offset[place_list[i]] = np.zeros_like(onset[place_list[i]])
            offset_t[place_list[i]] = np.zeros_like(onset[place_list[i]]).astype(np.float64)
            peak[place_list[i]] = np.zeros_like(onset[place_list[i]])
            peak_t[place_list[i]] = np.zeros_like(onset[place_list[i]]).astype(np.float64)
            for j in range(len(onset[place_list[i]])):
                offset[place_list[i]][j] = np.argmin(np.diff(temp, axis=1)[[j], -forward + 150:backward + 150],
                                                     axis=1)
                peak[place_list[i]][j] = np.argmax(temp[[j], -forward + 150:backward + 150], axis=1)
                offset_t[place_list[i]][j] = (np.argmin(np.diff(temp, axis=1)[[j], -forward + 150:backward + 150],
                                                        axis=1) - forward) / num
                peak_t[place_list[i]][j] = (np.argmax(temp[[j], -forward + 150:backward + 150],
                                                      axis=1) - forward) / num
        task_dict[task] = {'onset': onset_t, 'peak': peak_t, 'offset': offset_t}
        if method == 'plot1':
            fig, axs = plt.subplots(len(place_list)-len(blank_list), figsize=(15, 15))
            fig.suptitle(task, fontsize=16)
            blank_num = 0
            for place_ind in range(len(place_list)):

                if place_ind in blank_list:
                    blank_num+=1
                    continue
                temp = avgECoG[task][place_list[place_ind]][onset[place_list[place_ind]].argsort()]
                # temp[temp<0]=0
                axs[place_ind-blank_num].set_title(place_list[place_ind])
                sns.heatmap(temp, cmap="binary", ax=axs[place_ind-blank_num], vmax=3, vmin=0)
                xonsetplot = onset[place_list[place_ind]][onset[place_list[place_ind]].argsort()] +150- forward
                xoffsetplot = offset[place_list[place_ind]][onset[place_list[place_ind]].argsort()] +150- forward
                xpeakplot = peak[place_list[place_ind]][onset[place_list[place_ind]].argsort()] +150- forward

                yforplot = np.arange(1, len(peak[place_list[place_ind]]) + 1) - 0.5

                axs[place_ind-blank_num].scatter(xpeakplot, yforplot, c="b", alpha=0.4, s=0.01)
                axs[place_ind-blank_num].scatter(xoffsetplot, yforplot, c="r", alpha=0.4, s=0.01)
                axs[place_ind-blank_num].plot(xonsetplot, yforplot, "k--", alpha=0.4)
                axs[place_ind-blank_num].set_xticks([100,150,200,300])
            plt.xticks([100,150,200,300],[100,150,200,300])

            plt.show()
        else:
            for key in task_dict[task].keys():
                temp1 = task_dict[task][key]
                y_tick = temp1.keys()
                fig, ax = plt.subplots(figsize=(10, 10))
                values = [value for value in temp1.values()]
                sns.boxplot(data=values, orient='horizontal', width=0.3, flierprops={'markersize': 1})
                ax.set_title(f'{task}' + '_' + f'{key}')
                ax.set_yticklabels(y_tick, fontsize=8)
                ax.set_xlabel('Latency(s)')
                plt.show()


if __name__ == '__main__':
    HS_list = [44, 45, 47, 48, 50, 54, 71, 73, 76,78]

    # clean_path_data= "E:/vs/python/data_for_code"
    clean_data_path = "D:/BaiduSyncdisk/code"
    pairs_sound_avg(75,[45],clean_data_path)
    # method1='plot1'
    # method2='plot2'
    # lag_plot(HS_list, clean_path_data,method1)
    # lag_plot(HS_list, clean_path_data, method2)
    # "E:\\vs\\python\\data_for_code"