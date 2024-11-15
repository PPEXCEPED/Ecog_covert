#!/Users/DELL/anaconda3/envs/ECOG/python.exe
# -*- coding: utf-8 -*-
# @Time : 2023/11/4 21:46
# @Author : Zhenjie Wang
import numpy as np
import tqdm

from HS_reading import *

import datetime
import multiprocessing as mp




def run_exclude_feature(HS_list, model, clean_data_path,ex_features, feature_pool, task_name="covert",permutation_num = 200):
    """"
    ex_feature 是个列表，生成里面所有的delta
    feature_pool 是feature计算的总体
    """
    HS = HS_list[0]
    if HS <70 and task_name == "cue":
        return 0
    ex_features = ex_features[model]
    feature_pool = feature_pool[model]
    save_p_path = clean_data_path + "/TRF/" + "HS" + str(HS) + "/permutation/"
    save_path = clean_data_path + "/TRF/" + "HS" + str(HS) + "/"
    if not os.path.exists(save_p_path):
        os.mkdir(save_p_path)
    mat_file_name = save_path + "HS" + str(HS) + "_" + task_name + "_" + "_feat_mat.npy"

    if os.path.exists(mat_file_name):
        feat_mat = np.load(mat_file_name, allow_pickle=True).item()
    else:
        feat_mat = get_feat(HS, task_name)
        np.save(mat_file_name, feat_mat)

    nn_features_with_all = ["all"]
    for ex_feature in ex_features:
        feat_mat["ex_" + ex_feature] = []
        nn_features_with_all.append("ex_" + ex_feature)

        for i in range(len(feature_pool)):
            if feature_pool[i] != ex_feature:
                if len(feat_mat["ex_" + ex_feature]) == 0:
                    feat_mat["ex_" + ex_feature] = feat_mat[feature_pool[i]]
                else:
                    feat_mat["ex_" + ex_feature] = np.concatenate(
                        (feat_mat["ex_" + ex_feature], feat_mat[feature_pool[i]]), axis=1)
    feat_mat["all"] = []
    for i in range(len(feature_pool)):

        if len(feat_mat["all"]) == 0:
            feat_mat["all"] = feat_mat[feature_pool[i]]
        else:
            feat_mat["all"] = np.concatenate((feat_mat["all"], feat_mat[feature_pool[i]]), axis=1)

    fs = 10
    ds_a = -30
    ds_p = 30
    delays = np.arange(ds_a, ds_p)



    for i in tqdm.trange(permutation_num):
        r2_channel = {}
        r_channel = {}

        file_name = f"HS{HS}_hg_shuffle_full_{task_name}_{i}.npy"
        resp = np.load(save_p_path+"shuffle_data/"+file_name)
        for nn_feat_name in nn_features_with_all:
            dstim = feat_mat[nn_feat_name]

            test_corr_folds, wts_folds, best_alphas = run_cv_temporal_ridge_regression_model(
                dstim, resp, delays=delays, pred=False)

            r2 = np.sum(test_corr_folds ** 2, axis=0) / test_corr_folds.shape[0]
            r = np.mean(test_corr_folds, axis=0)

            r2_channel[nn_feat_name] = r2
            r_channel[nn_feat_name] = r


        np.save(save_path + "HS" + str(HS) + "_" + task_name + "_" + str(int(ds_a * fs)) + str(
            int(ds_p * fs)) + "_" + model + f"_r2_channel_{i}.npy", r2_channel)
        np.save(save_path + "HS" + str(HS) + "_" + task_name + "_" + str(int(ds_a * fs)) + str(
            int(ds_p * fs)) + "_" + model + f"_corr_channel_{i}.npy", r_channel)

    return 0

def generate_shuffled_data(HS, task_name="covert",permutation_num = 200):
    """"
    ex_feature 是个列表，生成里面所有的delta
    feature_pool 是feature计算的总体
    """
    save_p_path = clean_data_path + "TRF/" + "HS" + str(HS) + "/permutation/"
    save_path = clean_data_path + "TRF/" + "HS" + str(HS) + "/"
    if not os.path.exists(save_p_path):
        os.mkdir(save_p_path)
    for subfold in ["shuffle_data/","r/","r2/"]:
        if not os.path.exists(save_p_path+subfold):
            os.mkdir(save_p_path+subfold)

    mat_file_name = save_path + "HS" + str(HS) + "_" + task_name + "_" + "_feat_mat.npy"
    if os.path.exists(mat_file_name):
        feat_mat = np.load(mat_file_name, allow_pickle=True).item()
    else:
        feat_mat = get_feat(HS, task_name," ")
        np.save(mat_file_name, feat_mat)


    for i in tqdm.trange(permutation_num):

        np.random.seed(i)
        resp = copy.deepcopy(feat_mat["hg"])
        resp = np.random.shuffle(resp)
        file_name = f"HS{HS}_hg_shuffle_full_{i}.npy"
        np.save(save_p_path+"shuffle_data/"+file_name,resp)


if __name__ == '__main__':
    clean_data_path = "E:/DATA_Wangzhenjie/covert"
    # clean_data_path = "E:/DATA_Wangzhenjie/covert"
    start_t = datetime.datetime.now()
    feature_pool = {}
    ex_features = {}
    feature_pool["phonetic_model"] = ['bilabial', 'secondary labial', 'alveolar', 'velar',
                                      'mandibular', 'voiced', 'oral stop', 'fricative',
                                      'back tongue', 'high tongue', 'lip rounding',
                                      'jaw open']
    ex_features["phonetic_model"] = ['bilabial', 'secondary labial', 'alveolar', 'velar',
                                     'mandibular', 'voiced', 'oral stop', 'fricative',
                                     'back tongue', 'high tongue', 'lip rounding',
                                     'jaw open']

    # feature_pool["at_model"] = ["relPitch","lower_incisor","tongue","lip"]
    # ex_features["at_model"] = ["relPitch","lower_incisor","tongue","lip"]

    feature_pool["trace_model"] = ["relPitch", "lower_incisor", "upper_lip", "lower_lip", "tongue_tip", "tongue_body",
                                   "tongue_dorsum"]
    ex_features["trace_model"] = ["relPitch", "lower_incisor", "upper_lip", "lower_lip", "tongue_tip", "tongue_body",
                                  "tongue_dorsum"]

    feature_pool["audio_model"] = ['syllable', 'intensity', 'absPitch', 'f1', 'f2', 'peakRate']
    ex_features["audio_model"] = ['syllable', 'intensity', 'absPitch', 'f1', 'f2', 'peakRate']

    feature_pool["phonetic_model_2"] = ['consonantal_place', 'consonantal_manner', 'vowel_place']
    ex_features["phonetic_model_2"] = ['consonantal_place', 'consonantal_manner', 'vowel_place']

    num_cores = int(mp.cpu_count()) - 2
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores)
    param_dict = {
                    'task1': [44],
                  #    'task5': [45],
                  #  'task2': [47],
                  #  'task6': [48],
                  #  'task3': [50],
                  # 'task7': [54],
                  # 'task4': [71],
                  # 'task8': [73],
                  # 'task9': [76],
                  # 'task10': [78]
        }
    results = [pool.apply_async(run_exclude_feature, args=(param,"trace_model",clean_data_path,ex_features,feature_pool,"covert")) for name, param in param_dict.items()]
    results = [p.get() for p in results]

    # HS_list = [44,45,47,48,50,54,71,73,76,78]
    # for HS in HS_list:
    #     for task in ['overt','covert']:
    #         generate_shuffled_data(HS,task)
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("多进程计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")