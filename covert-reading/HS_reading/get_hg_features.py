import numpy as np
import scipy.io as scio
import os

def get_title():
    Ecog_title = "ECoG_"
    wav_title = "wav_"
    env_title = "env_"
    sound_title = "soundClipMat_"
    return Ecog_title,wav_title,env_title,sound_title

def get_hg_feat(HS, task, clean_data_path):
    """
    :param clean_data_path: 文件系统路径
    :param HS: 被试编好
    :param task_list: 要生成的featuremat任务列表
    :return: hg_feat
    """
    Ecog_title,wav_title,env_title,sound_title = get_title()
    if HS < 70:
        sound_list = ["ba", "bu", "da", "du", "ga", "gu"]
        forward = int(25)
        backward = int(85)
    else:
        sound_list = ["ba", "da", "ga", "pa", "ta", "ka", "sa", "sha"]
        forward = int(50)
        backward = int(150)
    onset_time = {"HS44": 0, "HS45": 0, "HS47": 0, "HS48": 0, "HS50": 0, "HS54": 0, "HS71": 0, "HS73": 0, "HS76": 0}

    mel_dir = clean_data_path+"mel_dir//"
    mel_feat_name = mel_dir + "HS" + str(HS) + "mel.npy"
    z = np.load(mel_feat_name, allow_pickle=True).item()

    at_dir = clean_data_path+"at_dir//"
    at_feat_name = at_dir + "HS" + str(HS) + "trace.npy"
    z2 = np.load(at_feat_name, allow_pickle=True).item()

    file_name = "HS" + str(HS) + "_Block_overt_covert.mat"
    z3 = scio.loadmat(os.path.join(clean_data_path+"/HSblockdata", file_name))

    z3 = z3["Alldata"][0][0]

    onset = int(np.floor(100 * onset_time["HS" + str(HS)])) + 150
#按照task将脑电数据进行合并，生成task_hg字典。task_hg字典的键为三个任务+音节，里面是完整的脑电数据
    if task == task:
        hg_feat=np.array([])
        for sound in sound_list:
            for i in range(z3[Ecog_title + task + "_" + sound].shape[0]):
                if len(hg_feat) == 0:
                    hg_feat = z3[Ecog_title + task + "_" + sound][i, :,onset - forward - 1:onset + backward - 1].T
                else:
                    hg_feat = np.concatenate((hg_feat,z3[Ecog_title + task + "_" + sound][i, :, onset - forward - 1:onset + backward - 1].T),axis=0)
    return hg_feat#将音节拼起来的矩阵



if __name__ == '__main__':

    clean_data_path = "E:/vs/python/data_for_code/"
    task='overt'
    HS=54

    feature_hg = get_hg_feat(HS,task,clean_data_path=clean_data_path)
    print(feature_hg.shape)