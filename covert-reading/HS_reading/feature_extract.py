#!/Users/DELL/anaconda3/envs/ECOG/python.exe
# -*- coding: utf-8 -*-
# @Time : 2023/9/23 15:29
# @Author : Zhenjie Wang
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch

def get_n_octaves(hz1, hz2):
    n = (np.log2(hz2) - np.log2(hz1))
    return n


def get_pitch_change(hz_values):
    hz_smooth = pd.Series(hz_values).rolling(center=True, window=5, win_type="hamming").mean()
    hz1 = np.copy(hz_smooth.values)[0:-1]
    hz2 = np.copy(hz_smooth.values)[1:]
    n = get_n_octaves(hz1, hz2)
    pitch_change = np.concatenate([[np.nan], n])
    return pitch_change


def slice2bin(x, timeline):
    timeline = np.array(timeline).squeeze(0)

    # 线性差值补全timeline里为0的点
    a = np.where(x[timeline] == 0)
    a = np.array(a).squeeze(0)
    total_seg = find_continus(a)
    inter_plote_num = 2
    for jj in range(len(total_seg)):
        inputx = x[total_seg[jj][0] + timeline[0] - 1:total_seg[jj][-1] + timeline[0] + inter_plote_num + 1]
        x[total_seg[jj][0] + timeline[0] - 1:total_seg[jj][-1] + timeline[0] + inter_plote_num + 1] = np.linspace(
            inputx[0], inputx[-1], len(inputx))

    return x


def find_continus(aa):
    l1 = []
    total = []
    for x in sorted(set(aa)):
        l1.append(x)
        if x + 1 not in aa:
            if 0 not in l1 and aa[-1] not in l1:
                total.append(l1)
            l1 = []
    return total


def get_transfer_onset_offset(HS, HS_block):
    onset_time = {"HS44": 0, "HS45": 0, "HS47": 0, "HS48": 0, "HS50": 0, "HS54": 0, "HS71": 0, "HS73": 0, "HS76": 0,
                  "HS78": 0}
    onset = int(np.floor(100 * onset_time["HS" + str(HS)])) + 150
    Ecog_title, wav_title, env_title, sound_title = get_title()

    print(HS)
    if HS < 70:
        sound_list_phoneme = ["b", "d", "g", "a", "u"]
        sound_list = ["ba", "da", "ga", "bu", "du", "gu"]
        forward = int(25)
        backward = int(85)
    else:
        sound_list_phoneme = ["b", "d", "g", "p", "t", "k", "s", "sh", "a"]
        sound_list = ["ba", "da", "ga", "pa", "ta", "ka", "sa", "sha"]
        forward = int(50)
        backward = int(150)
    transfer_delay = {}
    transfer_offset = {}
    for sound in sound_list:

        transfer_delay[sound] = []
        transfer_offset[sound] = []
        for i in range(HS_block[wav_title + "overt_" + sound][:, 1, onset - forward:onset + backward].shape[0]):
            transfer_delay[sound].append(np.where(HS_block[wav_title + "overt_" + sound][i, 1,
                                                  onset - forward:onset + backward] >= sound_list_phoneme.index(
                sound[-1]) + 1)[0][0])
            transfer_offset[sound].append(np.where(HS_block[wav_title + "overt_" + sound][i, 1,
                                                   onset - forward:onset + backward] >= sound_list_phoneme.index(
                sound[-1]) + 1)[0][-1])
        for i in range(len(transfer_delay[sound]), 84):
            transfer_delay[sound].append(np.NaN)
        for i in range(len(transfer_offset[sound]), 84):
            transfer_offset[sound].append(np.NaN)

    return pd.DataFrame(transfer_delay), pd.DataFrame(transfer_offset)


import scipy.io as scio
import os

n_padding_feature = {"syllable": 1, "phoneme": 1, "intensity": 1, "f1": 1, "f2": 1, "absPitch": 1, "relPitch": 1,
                     "drelPitch": 1, "Env": 1, "peakRate": 1, "peakEnv": 1, "mel": 80, "trace": 13, 'bilabial': 1,
                     'secondary labial': 1, 'alveolar': 1, 'velar': 1, 'voiced': 1, 'mandibular': 1, 'oral stop': 1,
                     'fricative': 1, 'back tongue': 1, 'high tongue': 1, 'lip rounding': 1, 'jaw open': 1,
                     "phonetics": 12, "consonantal_place": 5, "consonantal_manner": 3, "vowel_place": 4,
                     "ex_consonantal_place": 7, "ex_consonantal_manner": 9, "ex_vowel_place": 8, "hg": 256}


def get_phonetic_feature_cons(x, clean_data_path):
    phonetic_matrix = pd.read_csv(clean_data_path + "phonetic feature.csv", index_col="phonetic feature")
    phonetic_matrix = phonetic_matrix.values
    phonetic_matrix = np.nan_to_num(np.array(phonetic_matrix))
    sound_list_one = ["b", "d", "g", "b", "d", "g", "p", "t", "k", "s", "sh"]
    ind = sound_list_one.index(x[:-1])
    return phonetic_matrix[:, ind]


def padding_ds(ddict, padding_space=60):
    pass
    # for i in list(ddict.keys()):
    #     if i in list(n_padding_feature.keys()):
    #         if len(ddict[i])==0:
    #             ddict[i] = np.zeros([padding_space,n_padding_feature[i]])
    #         else:
    #             if n_padding_feature[i] == 1:
    #                 ddict[i] = np.expand_dims(ddict[i],-1)
    #             ddict[i] = np.concatenate((ddict[i],np.zeros([padding_space,n_padding_feature[i]])),axis=0)
    #         if n_padding_feature[i] == 1:
    #            ddict[i] = ddict[i].squeeze(-1)
    return ddict


def convert_sequence(sequence):
    sequence = sequence.tolist()
    max_value = max(sequence)
    converted_sequence = [1 if value == max_value else 0 for value in sequence]
    return np.expand_dims(np.array(converted_sequence), -1)


def init_feat():
    feat_mat = {}
    feat_wav_list = ["syllable", "phoneme", "intensity", "f1", "f2"]
    for i in feat_wav_list:
        feat_mat[i] = []
    feat_mat['syllable_onehot'] = []
    feat_mat['phoneme_onehot'] = []
    feat_env_list = ["Env", "peakRate", "peakEnv"]
    for i in feat_env_list:
        feat_mat[i] = []
    feat_pitch_list = ["absPitch_bin", "relPitch_bin", "drelPitch_bin"]
    for i in feat_pitch_list:
        feat_mat[i] = []
    feat_phonetic_list = ['bilabial', 'secondary labial', 'alveolar', 'velar', 'mandibular',
                          'voiced', 'oral stop', 'fricative', 'back tongue', 'high tongue',
                          'lip rounding', 'jaw open']

    feat_mat["mel"] = []
    feat_mat["trace"] = []
    feat_mat["hg"] = []

    feat_mat["absPitch"] = []
    feat_mat["relPitch"] = []
    feat_mat["drelPitch"] = []

    for i in feat_phonetic_list:
        feat_mat[i] = []

    feat_mat['phonetics'] = []
    feat_mat['consonantal_place'] = []
    feat_mat['consonantal_manner'] = []
    feat_mat['vowel_place'] = []

    return feat_mat


def get_title():
    Ecog_title = "ECoG_"
    wav_title = "wav_"
    env_title = "env_"
    sound_title = "soundClipMat_"
    return Ecog_title, wav_title, env_title, sound_title



def get_feat(HS, task, clean_data_path):
    """

    :param clean_data_path: 文件系统路径
    :param HS: 被试编好
    :param task: 要生成的featuremat任务
    :return: n_timepoints,n_feature
    """
    Ecog_title, wav_title, env_title, sound_title = get_title()
    if HS < 70:
        sound_list = ["ba", "bu", "da", "du", "ga", "gu"]
        forward = int(25)
        backward = int(85)
        phoneme_onehot = ["b", "d", "g", "a", "u"]
    else:
        sound_list = ["ba", "da", "ga", "pa", "ta", "ka", "sa", "sha"]
        forward = int(50)
        backward = int(150)
        phoneme_onehot = ["b", "d", "g", "p", "t", "k", "s", "sh", "a"]

    onset_time = {"HS44": 0, "HS45": 0, "HS47": 0, "HS48": 0, "HS50": 0, "HS54": 0, "HS71": 0, "HS73": 0, "HS76": 0,
                  "HS78": 0}

    mel_dir = clean_data_path + "mel_dir/"
    mel_feat_name = mel_dir + "HS" + str(HS) + "mel.npy"
    z = np.load(mel_feat_name, allow_pickle=True).item()
    at_dir = clean_data_path + "at_dir/"
    at_feat_name = at_dir + "HS" + str(HS) + "trace.npy"
    z2 = np.load(at_feat_name, allow_pickle=True).item()
    file_name = "HS" + str(HS) + "_Block_overt_covert.mat"
    z3 = scio.loadmat(os.path.join(clean_data_path + "/HSblockdata", file_name))
    z3 = z3["Alldata"][0][0]
    feat_mat = init_feat()
    feat_wav_list = ["syllable", "phoneme", "intensity", "f1", "f2"]
    feat_env_list = ["Env", "peakRate", "peakEnv"]
    feat_pitch_bin_list = ["absPitch_bin", "relPitch_bin", "drelPitch_bin"]
    feat_pitch_list = ["absPitch", "relPitch", "drelPitch"]
    feat_phonetic_list = ['bilabial', 'secondary labial', 'alveolar', 'velar', 'mandibular',
                          'voiced', 'oral stop', 'fricative', 'back tongue', 'high tongue',
                          'lip rounding', 'jaw open']

    onset = int(np.floor(100 * onset_time["HS" + str(HS)])) + 150

    if task == "overt":

        for sound in sound_list:

            pitch_set = np.zeros([z3[Ecog_title + task + "_" + sound].shape[0], 3, 400])
            feature_phonetic_index = z3[wav_title + task + "_" + sound].shape[1] - 12
            print(sound)
            for i in range(z3[Ecog_title + task + "_" + sound].shape[0]):


                feat_mat = padding_ds(feat_mat)
                # 减一是因为mel和trace是398个timepoint数据
                if len(feat_mat["mel"]) == 0:
                    feat_mat["mel"] = z[sound_title + task + "_" + sound][i,
                                      onset - forward - 1:onset + backward - 1, :]
                else:
                    feat_mat["mel"] = np.concatenate(
                        (feat_mat["mel"],
                         z[sound_title + task + "_" + sound][i, onset - forward - 1:onset + backward - 1, :]),
                        axis=0)

                if len(feat_mat["hg"]) == 0:
                    feat_mat["hg"] = z3[Ecog_title + task + "_" + sound][i, :,
                                     onset - forward - 1:onset + backward - 1].T
                else:
                    feat_mat["hg"] = np.concatenate(
                        (feat_mat["hg"],
                         z3[Ecog_title + task + "_" + sound][i, :, onset - forward - 1:onset + backward - 1].T),
                        axis=0)

                for wav_feature in range(len(feat_wav_list)):
                    temp = z3[wav_title + task + "_" + sound][i, wav_feature, onset - forward:onset + backward].T

                    if wav_feature < 2:  # 针对onset进行混淆,a 和辅音有不同的值，diff后a>辅音，辅音>0
                        temp = np.diff(temp)
                        temp = np.concatenate((np.array([0]), temp), axis=0)
                        temp[temp > 0] = 1

                    if len(feat_mat[feat_wav_list[wav_feature]]) == 0:
                        feat_mat[feat_wav_list[wav_feature]] = temp
                    else:
                        feat_mat[feat_wav_list[wav_feature]] = np.concatenate(
                            (feat_mat[feat_wav_list[wav_feature]], temp), axis=0)
                # 对one_hot的处理，
                syllable_onehot_slice = F.one_hot(torch.LongTensor(z3[wav_title + task + "_" + sound][i, 0, onset - forward:onset + backward]), len(sound_list)+1).numpy()
                phoneme_onehot_slice = F.one_hot(torch.LongTensor(z3[wav_title + task + "_" + sound][i, 1, onset - forward:onset + backward]), len(phoneme_onehot)+1).numpy()

                if len(feat_mat['syllable_onehot']) == 0:
                    feat_mat['syllable_onehot'] = syllable_onehot_slice[:,1:]
                else:
                    feat_mat['syllable_onehot'] = np.concatenate(
                        (feat_mat['syllable_onehot'], syllable_onehot_slice[:,1:]), axis=0)
                if len(feat_mat['phoneme_onehot']) == 0:
                    feat_mat['phoneme_onehot'] = phoneme_onehot_slice[:,1:]
                else:
                    feat_mat['phoneme_onehot'] = np.concatenate(
                        (feat_mat['phoneme_onehot'], phoneme_onehot_slice[:,1:]), axis=0)

                for env_feature in range(len(feat_env_list)):
                    temp = z3[env_title + task + "_" + sound][i, env_feature,
                           onset - forward:onset + backward].T
                    if env_feature > 1:
                        temp = convert_sequence(temp)

                    if len(feat_mat[feat_env_list[env_feature]]) == 0:
                        feat_mat[feat_env_list[env_feature]] = temp
                    else:
                        feat_mat[feat_env_list[env_feature]] = np.concatenate((feat_mat[feat_env_list[env_feature]],
                                                                               temp),
                                                                              axis=0)

                for pitch_feature in range(len(feat_pitch_bin_list)):
                    feature_two_index = 5
                    temp = np.nan_to_num(z3[wav_title + task + "_" + sound][i,
                                         feature_two_index + pitch_feature * 10:feature_two_index + pitch_feature * 10 + 10,
                                         onset - forward:onset + backward])

                    if len(feat_mat[feat_pitch_bin_list[pitch_feature]]) == 0:
                        feat_mat[feat_pitch_bin_list[pitch_feature]] = temp.T
                    else:
                        feat_mat[feat_pitch_bin_list[pitch_feature]] = np.concatenate(
                            (feat_mat[feat_pitch_bin_list[pitch_feature]], temp.T), axis=0)
                pitches = np.nan_to_num(z3[wav_title + task + "_" + sound][i, 35,
                                        onset - forward:onset + backward].T)
                if len(feat_mat['absPitch']) == 0:
                    feat_mat['absPitch'] = pitches
                else:
                    feat_mat['absPitch'] = np.concatenate(
                        (feat_mat['absPitch'], pitches), axis=0)
                rel_pitches = np.nan_to_num(z3[wav_title + task + "_" + sound][i, 36,
                                            onset - forward:onset + backward].T)

                if len(feat_mat['relPitch']) == 0:
                    feat_mat['relPitch'] = rel_pitches
                else:
                    feat_mat['relPitch'] = np.concatenate(
                        (feat_mat['relPitch'], rel_pitches), axis=0)
                pitch_set[i, 0, onset - forward:onset + backward] = pitches
                pitch_set[i, 1, onset - forward:onset + backward] = rel_pitches
                pitches[pitches == 0] = np.NaN
                pitch_change = np.nan_to_num(get_pitch_change(np.exp(pitches)) * 100)
                pitch_set[i, 2, onset - forward:onset + backward] = pitch_change

                if len(feat_mat['drelPitch']) == 0:
                    feat_mat['drelPitch'] = pitch_change
                else:
                    feat_mat['drelPitch'] = np.concatenate(
                        (feat_mat['drelPitch'], pitch_change), axis=0)
                if len(feat_mat["trace"]) == 0:
                    feat_mat["trace"] = np.concatenate((z2[sound_title + task + "_" + sound][i,
                                                        onset - forward - 1:onset + backward - 1, :],
                                                        np.expand_dims(rel_pitches, -1)), axis=1)
                else:
                    feat_mat["trace"] = np.concatenate(
                        (feat_mat["trace"], np.concatenate((z2[sound_title + task + "_" + sound][i,
                                                            onset - forward - 1:onset + backward - 1, :],
                                                            np.expand_dims(rel_pitches, -1)), axis=1)), axis=0)

                for phonetic_feature in range(len(feat_phonetic_list)):
                    temp = z3[wav_title + task + "_" + sound][i, feature_phonetic_index + phonetic_feature,
                           onset - forward:onset + backward].T

                    if len(feat_mat[feat_phonetic_list[phonetic_feature]]) == 0:
                        feat_mat[feat_phonetic_list[phonetic_feature]] = temp.T
                    else:
                        feat_mat[feat_phonetic_list[phonetic_feature]] = np.concatenate(
                            (feat_mat[feat_phonetic_list[phonetic_feature]], temp.T), axis=0)

                if len(feat_mat['phonetics']) == 0:
                    feat_mat['phonetics'] = z3[wav_title + task + "_" + sound][i, feature_phonetic_index:,
                                            onset - forward:onset + backward].T
                else:
                    feat_mat['phonetics'] = np.concatenate(
                        (feat_mat['phonetics'], z3[wav_title + task + "_" + sound][i, feature_phonetic_index:,
                                                onset - forward:onset + backward].T), axis=0)
                if len(feat_mat['consonantal_place']) == 0:
                    feat_mat['consonantal_place'] = z3[wav_title + task + "_" + sound][i,
                                                    feature_phonetic_index:feature_phonetic_index + 5,
                                                    onset - forward:onset + backward].T
                else:
                    feat_mat['consonantal_place'] = np.concatenate(
                        (feat_mat['consonantal_place'],
                         z3[wav_title + task + "_" + sound][i, feature_phonetic_index:feature_phonetic_index + 5,
                         onset - forward:onset + backward].T), axis=0)
                if len(feat_mat['consonantal_manner']) == 0:
                    feat_mat['consonantal_manner'] = z3[wav_title + task + "_" + sound][i,
                                                     feature_phonetic_index + 5:feature_phonetic_index + 8,
                                                     onset - forward:onset + backward].T
                else:
                    feat_mat['consonantal_manner'] = np.concatenate(
                        (feat_mat['consonantal_manner'], z3[wav_title + task + "_" + sound][i,
                                                         feature_phonetic_index + 5:feature_phonetic_index + 8,
                                                         onset - forward:onset + backward].T), axis=0)
                if len(feat_mat['vowel_place']) == 0:
                    feat_mat['vowel_place'] = z3[wav_title + task + "_" + sound][i,
                                              feature_phonetic_index + 8:feature_phonetic_index + 12,
                                              onset - forward:onset + backward].T
                else:
                    feat_mat['vowel_place'] = np.concatenate(
                        (feat_mat['vowel_place'], z3[wav_title + task + "_" + sound][i,
                                                  feature_phonetic_index + 8:feature_phonetic_index + 12,
                                                  onset - forward:onset + backward].T), axis=0)
                feat_mat = padding_ds(feat_mat)
            save_pitch_path = clean_data_path + "TRF/HS" + str(HS) + "/HS" + str(HS) + sound + ".npy"
            np.save(save_pitch_path, pitch_set)

    if task == "covert":
        transfer_point_list, offset_list = get_transfer_onset_offset(HS, z3)
        transfer_point_list = np.around(np.nanmean(transfer_point_list, axis=0))
        offset_list = np.around(np.nanmean(offset_list, axis=0))
        print(len(transfer_point_list), len(offset_list))
        for sound in sound_list:

            tempz = np.mean(z[sound_title + "overt_" + sound][:, onset - forward - 1:onset + backward - 1, :], axis=0)

            transfer_point = int(transfer_point_list[sound_list.index(sound)])
            offset = int(offset_list[sound_list.index(sound)])

            temp_onset = np.zeros([backward + forward])
            temp_onset[forward] = 1
            temp_offset = np.zeros([backward + forward])
            temp_offset[forward] = 1
            temp_offset[transfer_point] = 1

            syllable_onehot_slice = np.zeros([backward + forward,len(sound_list)])
            phoneme_onehot_slice = np.zeros([backward + forward,len(phoneme_onehot)])

            syllable_onehot_slice[forward,sound_list.index(sound)] = 1
            phoneme_onehot_slice[forward,phoneme_onehot.index(sound[:-1])] = 1
            phoneme_onehot_slice[transfer_point, phoneme_onehot.index(sound[-1])] = 1


            phonetic_feature_temp = np.expand_dims(get_phonetic_feature_cons(sound, clean_data_path), -1)
            phonetic_feature_avg = np.zeros([12, backward + forward])
            phonetic_feature_avg[:8, forward:transfer_point] = np.repeat(phonetic_feature_temp[:8],
                                                                         transfer_point - forward, axis=1)
            phonetic_feature_avg[8:12, transfer_point:offset] = np.repeat(phonetic_feature_temp[8:],
                                                                          offset - transfer_point,
                                                                          axis=1)

            wav_avg = np.mean(z3[wav_title + "overt_" + sound][:, :5, onset - forward:onset + backward], axis=0)
            wav_avg[0, :] = temp_onset
            wav_avg[1, :] = temp_offset
            wav_avg[3, :] = np.log(
                np.mean(np.exp(z3[wav_title + "overt_" + sound][:, 3, onset - forward:onset + backward]), axis=0))
            wav_avg[4, :] = np.log(
                np.mean(np.exp(z3[wav_title + "overt_" + sound][:, 4, onset - forward:onset + backward]), axis=0))

            env_avg = np.mean(z3[env_title + "overt_" + sound][:, :, onset - forward:onset + backward], axis=0)
            env_avg[[1], :] = convert_sequence(env_avg[1, :]).T
            env_avg[[2], :] = convert_sequence(env_avg[2, :]).T

            pitch_set = np.load(clean_data_path + "TRF/HS" + str(HS) + "/HS" + str(HS) + sound + ".npy")
            pitch_avg = np.mean(pitch_set, axis=0)
            pitch_avg[0, :] = np.log(np.mean(np.exp(pitch_set[:, 0, :]), axis=0))
            pitch_avg[1, :] = np.log(np.mean(np.exp(pitch_set[:, 1, :]), axis=0))

            trace_avg = np.mean(z2[sound_title + "overt_" + sound][:, onset - forward - 1:onset + backward - 1, :],
                                axis=0)

            trace_avg = np.concatenate((trace_avg, pitch_avg[[1], onset - forward:onset + backward].T), axis=1)

            for i in range(z3[Ecog_title + "covert_" + sound].shape[0]):
                feat_mat = padding_ds(feat_mat)
                if len(feat_mat["mel"]) == 0:
                    feat_mat["mel"] = tempz
                else:
                    feat_mat["mel"] = np.concatenate((feat_mat["mel"], tempz), axis=0)
                if len(feat_mat["trace"]) == 0:
                    feat_mat["trace"] = trace_avg
                else:
                    feat_mat["trace"] = np.concatenate((feat_mat["trace"], trace_avg), axis=0)
                if len(feat_mat["hg"]) == 0:
                    feat_mat["hg"] = z3[Ecog_title + "covert_" + sound][i, :, onset - forward:onset + backward].T
                else:
                    feat_mat["hg"] = np.concatenate(
                        (feat_mat["hg"], z3[Ecog_title + "covert_" + sound][i, :, onset - forward:onset + backward].T),
                        axis=0)

                for wav_feature in range(len(feat_wav_list)):

                    if len(feat_mat[feat_wav_list[wav_feature]]) == 0:
                        feat_mat[feat_wav_list[wav_feature]] = wav_avg[wav_feature].T
                    else:
                        feat_mat[feat_wav_list[wav_feature]] = np.concatenate(
                            (feat_mat[feat_wav_list[wav_feature]], wav_avg[wav_feature].T), axis=0)

                if len(feat_mat['syllable_onehot']) == 0:
                    feat_mat['syllable_onehot'] = syllable_onehot_slice
                else:
                    feat_mat['syllable_onehot'] = np.concatenate(
                        (feat_mat['syllable_onehot'], syllable_onehot_slice), axis=0)
                if len(feat_mat['phoneme_onehot']) == 0:
                    feat_mat['phoneme_onehot'] = phoneme_onehot_slice
                else:
                    feat_mat['phoneme_onehot'] = np.concatenate(
                        (feat_mat['phoneme_onehot'], phoneme_onehot_slice), axis=0)

                for env_feature in range(len(feat_env_list)):

                    if len(feat_mat[feat_env_list[env_feature]]) == 0:
                        feat_mat[feat_env_list[env_feature]] = env_avg[env_feature].T
                    else:
                        feat_mat[feat_env_list[env_feature]] = np.concatenate(
                            (feat_mat[feat_env_list[env_feature]], env_avg[env_feature].T), axis=0)

                for pitch_feature in range(len(feat_pitch_list)):

                    if len(feat_mat[feat_pitch_list[pitch_feature]]) == 0:
                        feat_mat[feat_pitch_list[pitch_feature]] = pitch_avg[pitch_feature,
                                                                   onset - forward:onset + backward].T
                    else:
                        feat_mat[feat_pitch_list[pitch_feature]] = np.concatenate(
                            (feat_mat[feat_pitch_list[pitch_feature]],
                             pitch_avg[pitch_feature, onset - forward:onset + backward].T), axis=0)

                for phonetic_feature in range(len(feat_phonetic_list)):
                    temp = phonetic_feature_avg[phonetic_feature]

                    if len(feat_mat[feat_phonetic_list[phonetic_feature]]) == 0:
                        feat_mat[feat_phonetic_list[phonetic_feature]] = temp.T
                    else:
                        feat_mat[feat_phonetic_list[phonetic_feature]] = np.concatenate(
                            (feat_mat[feat_phonetic_list[phonetic_feature]], temp.T), axis=0)

                if len(feat_mat['phonetics']) == 0:
                    feat_mat['phonetics'] = phonetic_feature_avg.T
                else:
                    feat_mat['phonetics'] = np.concatenate(
                        (feat_mat['phonetics'], phonetic_feature_avg.T), axis=0)
                if len(feat_mat['consonantal_place']) == 0:
                    feat_mat['consonantal_place'] = phonetic_feature_avg[:5].T
                else:
                    feat_mat['consonantal_place'] = np.concatenate(
                        (feat_mat['consonantal_place'],
                         phonetic_feature_avg[:5].T), axis=0)
                if len(feat_mat['consonantal_manner']) == 0:
                    feat_mat['consonantal_manner'] = phonetic_feature_avg[5:8].T
                else:
                    feat_mat['consonantal_manner'] = np.concatenate(
                        (feat_mat['consonantal_manner'], phonetic_feature_avg[5:8].T), axis=0)

                if len(feat_mat['vowel_place']) == 0:
                    feat_mat['vowel_place'] = phonetic_feature_avg[8:].T
                else:
                    feat_mat['vowel_place'] = np.concatenate(
                        (feat_mat['vowel_place'], phonetic_feature_avg[8:].T), axis=0)
                feat_mat = padding_ds(feat_mat)
    if task == "cue":
        transfer_point_list, offset_list = get_transfer_onset_offset(HS, z3)
        transfer_point_list = np.around(np.nanmean(transfer_point_list, axis=0))
        offset_list = np.around(np.nanmean(offset_list, axis=0))
        print(len(transfer_point_list), len(offset_list))

        for sound in sound_list:
            tempz = np.mean(z[sound_title + "overt_" + sound][:, onset - forward - 1:onset + backward - 1, :], axis=0)

            transfer_point = int(transfer_point_list[sound_list.index(sound)])
            offset = int(offset_list[sound_list.index(sound)])

            temp_onset = np.zeros([backward + forward])
            temp_onset[forward] = 1
            temp_offset = np.zeros([backward + forward])
            temp_offset[forward] = 1
            temp_offset[transfer_point] = 1

            syllable_onehot_slice = np.zeros([backward + forward,len(sound_list)])
            phoneme_onehot_slice = np.zeros([backward + forward,len(phoneme_onehot)])

            syllable_onehot_slice[forward,sound_list.index(sound)] = 1
            phoneme_onehot_slice[forward,phoneme_onehot.index(sound[:-1])] = 1
            phoneme_onehot_slice[transfer_point, phoneme_onehot.index(sound[-1])] = 1


            phonetic_feature_temp = np.expand_dims(get_phonetic_feature_cons(sound, clean_data_path), -1)
            phonetic_feature_avg = np.zeros([12, backward + forward])
            phonetic_feature_avg[:8, forward:transfer_point] = np.repeat(phonetic_feature_temp[:8],
                                                                         transfer_point - forward, axis=1)
            phonetic_feature_avg[8:12, transfer_point:offset] = np.repeat(phonetic_feature_temp[8:],
                                                                          offset - transfer_point,
                                                                          axis=1)

            wav_avg = np.mean(z3[wav_title + "overt_" + sound][:, :5, onset - forward:onset + backward], axis=0)
            wav_avg[0, :] = temp_onset
            wav_avg[1, :] = temp_offset
            wav_avg[3, :] = np.log(
                np.mean(np.exp(z3[wav_title + "overt_" + sound][:, 3, onset - forward:onset + backward]), axis=0))
            wav_avg[4, :] = np.log(
                np.mean(np.exp(z3[wav_title + "overt_" + sound][:, 4, onset - forward:onset + backward]), axis=0))

            env_avg = np.mean(z3[env_title + "overt_" + sound][:, :, onset - forward:onset + backward], axis=0)
            env_avg[[1], :] = convert_sequence(env_avg[1, :]).T
            env_avg[[2], :] = convert_sequence(env_avg[2, :]).T

            pitch_set = np.load(clean_data_path + "TRF/HS" + str(HS) + "/HS" + str(HS) + sound + ".npy")
            pitch_avg = np.mean(pitch_set, axis=0)
            pitch_avg[0, :] = np.log(np.mean(np.exp(pitch_set[:, 0, :]), axis=0))
            pitch_avg[1, :] = np.log(np.mean(np.exp(pitch_set[:, 1, :]), axis=0))

            trace_avg = np.mean(z2[sound_title + "overt_" + sound][:, onset - forward - 1:onset + backward - 1, :],
                                axis=0)

            trace_avg = np.concatenate((trace_avg, pitch_avg[[1], onset - forward:onset + backward].T), axis=1)

            for i in range(z3[Ecog_title + "cue_" + sound].shape[0]):
                feat_mat = padding_ds(feat_mat)
                if len(feat_mat["mel"]) == 0:
                    feat_mat["mel"] = tempz
                else:
                    feat_mat["mel"] = np.concatenate((feat_mat["mel"], tempz), axis=0)
                if len(feat_mat["trace"]) == 0:
                    feat_mat["trace"] = trace_avg
                else:
                    feat_mat["trace"] = np.concatenate((feat_mat["trace"], trace_avg), axis=0)
                if len(feat_mat["hg"]) == 0:
                    feat_mat["hg"] = z3[Ecog_title + "cue_" + sound][i, :, onset - forward:onset + backward].T
                else:
                    feat_mat["hg"] = np.concatenate(
                        (feat_mat["hg"], z3[Ecog_title + "cue_" + sound][i, :, onset - forward:onset + backward].T),
                        axis=0)

                for wav_feature in range(len(feat_wav_list)):

                    if len(feat_mat[feat_wav_list[wav_feature]]) == 0:
                        feat_mat[feat_wav_list[wav_feature]] = wav_avg[wav_feature].T
                    else:
                        feat_mat[feat_wav_list[wav_feature]] = np.concatenate(
                            (feat_mat[feat_wav_list[wav_feature]], wav_avg[wav_feature].T), axis=0)

                for env_feature in range(len(feat_env_list)):

                    if len(feat_mat[feat_env_list[env_feature]]) == 0:
                        feat_mat[feat_env_list[env_feature]] = env_avg[env_feature].T
                    else:
                        feat_mat[feat_env_list[env_feature]] = np.concatenate(
                            (feat_mat[feat_env_list[env_feature]], env_avg[env_feature].T), axis=0)

                for pitch_feature in range(len(feat_pitch_list)):

                    if len(feat_mat[feat_pitch_list[pitch_feature]]) == 0:
                        feat_mat[feat_pitch_list[pitch_feature]] = pitch_avg[pitch_feature,
                                                                   onset - forward:onset + backward].T
                    else:
                        feat_mat[feat_pitch_list[pitch_feature]] = np.concatenate(
                            (feat_mat[feat_pitch_list[pitch_feature]],
                             pitch_avg[pitch_feature, onset - forward:onset + backward].T), axis=0)

                if len(feat_mat['syllable_onehot']) == 0:
                    feat_mat['syllable_onehot'] = syllable_onehot_slice
                else:
                    feat_mat['syllable_onehot'] = np.concatenate(
                        (feat_mat['syllable_onehot'], syllable_onehot_slice), axis=0)
                if len(feat_mat['phoneme_onehot']) == 0:
                    feat_mat['phoneme_onehot'] = phoneme_onehot_slice
                else:
                    feat_mat['phoneme_onehot'] = np.concatenate(
                        (feat_mat['phoneme_onehot'], phoneme_onehot_slice), axis=0)
                for phonetic_feature in range(len(feat_phonetic_list)):
                    temp = phonetic_feature_avg[phonetic_feature]

                    if len(feat_mat[feat_phonetic_list[phonetic_feature]]) == 0:
                        feat_mat[feat_phonetic_list[phonetic_feature]] = temp.T
                    else:
                        feat_mat[feat_phonetic_list[phonetic_feature]] = np.concatenate(
                            (feat_mat[feat_phonetic_list[phonetic_feature]], temp.T), axis=0)

                if len(feat_mat['phonetics']) == 0:
                    feat_mat['phonetics'] = phonetic_feature_avg.T
                else:
                    feat_mat['phonetics'] = np.concatenate(
                        (feat_mat['phonetics'], phonetic_feature_avg.T), axis=0)
                if len(feat_mat['consonantal_place']) == 0:
                    feat_mat['consonantal_place'] = phonetic_feature_avg[:5].T
                else:
                    feat_mat['consonantal_place'] = np.concatenate(
                        (feat_mat['consonantal_place'],
                         phonetic_feature_avg[:5].T), axis=0)
                if len(feat_mat['consonantal_manner']) == 0:
                    feat_mat['consonantal_manner'] = phonetic_feature_avg[5:8].T
                else:
                    feat_mat['consonantal_manner'] = np.concatenate(
                        (feat_mat['consonantal_manner'], phonetic_feature_avg[5:8].T), axis=0)
                if len(feat_mat['vowel_place']) == 0:
                    feat_mat['vowel_place'] = phonetic_feature_avg[8:].T
                else:
                    feat_mat['vowel_place'] = np.concatenate(
                        (feat_mat['vowel_place'], phonetic_feature_avg[8:].T), axis=0)
                feat_mat = padding_ds(feat_mat)

    feat_mat["lower_incisor"] = feat_mat["trace"][:, :2]
    feat_mat["upper_lip"] = feat_mat["trace"][:, 2:4]
    feat_mat["lower_lip"] = feat_mat["trace"][:, 4:6]
    feat_mat["tongue_tip"] = feat_mat["trace"][:, 6:8]
    feat_mat["tongue_body"] = feat_mat["trace"][:, 8:10]
    feat_mat["tongue_dorsum"] = feat_mat["trace"][:, 10:12]
    #
    # feat_mat["diff_lower_incisor"] = np.concatenate((np.zeros([1, 2]), np.diff(feat_mat["lower_incisor"], axis=0)),
    #                                                 axis=0)
    # feat_mat["diff_upper_lip"] = np.concatenate((np.zeros([1, 2]), np.diff(feat_mat["upper_lip"], axis=0)), axis=0)
    # feat_mat["diff_lower_lip"] = np.concatenate((np.zeros([1, 2]), np.diff(feat_mat["lower_lip"], axis=0)), axis=0)
    # feat_mat["diff_tongue_tip"] = np.concatenate((np.zeros([1, 2]), np.diff(feat_mat["tongue_tip"], axis=0)), axis=0)
    # feat_mat["diff_tongue_body"] = np.concatenate((np.zeros([1, 2]), np.diff(feat_mat["tongue_body"], axis=0)), axis=0)
    # feat_mat["diff_tongue_dorsum"] = np.concatenate((np.zeros([1, 2]), np.diff(feat_mat["tongue_dorsum"], axis=0)),
    #                                                 axis=0)

    feat_mat["tongue"] = feat_mat["trace"][:, 6:12]
    feat_mat["lip"] = feat_mat["trace"][:, 2:6]

    # feat_mat["all"] = np.concatenate((feat_mat["mel"], feat_mat["trace"]), axis=1)

    for i in feat_wav_list:
        feat_mat[i] = np.expand_dims(feat_mat[i], -1)
    #
    #     feat_mat["all"] = np.concatenate((feat_mat["all"], feat_mat[i]), axis=1)
    #
    for i in feat_env_list:
        feat_mat[i] = np.expand_dims(feat_mat[i], -1)
    #     feat_mat["all"] = np.concatenate((feat_mat["all"], feat_mat[i]), axis=1)

    for i in feat_pitch_list:
        feat_mat[i] = np.expand_dims(feat_mat[i], -1)

    for i in feat_phonetic_list:
        feat_mat[i] = np.expand_dims(feat_mat[i], -1)
    # feat_mat["diff_relPitch"] = np.concatenate((np.zeros([1, 1]), np.diff(feat_mat["relPitch"], axis=0)), axis=0)

    return feat_mat


def get_hg_feat(HS, task, clean_data_path):
    """
    :param clean_data_path: 文件系统路径
    :param HS: 被试编好
    :param task_list: 要生成的featuremat任务列表
    :return: hg_feat
    """
    Ecog_title, wav_title, env_title, sound_title = get_title()
    if HS < 70:
        sound_list = ["ba", "bu", "da", "du", "ga", "gu"]
        forward = int(25)
        backward = int(85)
    else:
        sound_list = ["ba", "da", "ga", "pa", "ta", "ka", "sa", "sha"]
        forward = int(50)
        backward = int(150)
    onset_time = {"HS44": 0, "HS45": 0, "HS47": 0, "HS48": 0, "HS50": 0, "HS54": 0, "HS71": 0, "HS73": 0, "HS76": 0,
                  "HS78": 0}

    file_name = "HS" + str(HS) + "_Block_overt_covert.mat"
    z3 = scio.loadmat(os.path.join(clean_data_path + "/HSblockdata", file_name))

    z3 = z3["Alldata"][0][0]

    onset = int(np.floor(100 * onset_time["HS" + str(HS)])) + 150
    # 按照task将脑电数据进行合并，生成task_hg字典。task_hg字典的键为三个任务+音节，里面是完整的脑电数据

    hg_feat = np.array([])
    for sound in sound_list:
        for i in range(z3[Ecog_title + task + "_" + sound].shape[0]):
            if len(hg_feat) == 0:
                hg_feat = z3[Ecog_title + task + "_" + sound][i, :, onset - forward:onset + backward].T
            else:
                hg_feat = np.concatenate(
                    (hg_feat, z3[Ecog_title + task + "_" + sound][i, :, onset - forward:onset + backward].T), axis=0)
    return hg_feat  # 将音节拼起来的矩阵


if __name__ == '__main__':
    clean_data_path = "/Users/zhaozehao/Desktop/reading task/"
    feature = get_feat(54, task="covert", clean_data_path=clean_data_path)

    for keys in feature.keys():
        feat_pitch_bin_list = ["absPitch_bin", "relPitch_bin", "drelPitch_bin"]
        if keys not in feat_pitch_bin_list:
            print(keys, feature[keys].shape)
