import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from HS_reading.classification import classification
import copy

################################
# 本文件用于绘制各个脑区之间的lag latency


def prepare_dict(d, type):
    # 去重脑区
    copy_dict = copy.deepcopy(d)
    for pairs in copy_dict.keys():
        if type == 'r':
            if not np.any(np.isnan(np.array(copy_dict[pairs]))):
                pair_reverse = pairs.split('_')[1] + '_' + pairs.split('_')[0]
                if pair_reverse in d and pairs in d:
                    d[pairs] = np.concatenate((d[pairs], d[pair_reverse]), axis=0)
                    del d[pair_reverse]
                    copy_dict[pair_reverse][:] = np.nan
        else:
            if not np.any(np.isnan(np.array(copy_dict[pairs]))):
                pair_reverse = pairs.split('_')[1] + '_' + pairs.split('_')[0]
                if pair_reverse in d and pairs in d:
                    d[pairs] = np.concatenate((d[pairs], -d[pair_reverse]), axis=0)
                    del d[pair_reverse]
                    copy_dict[pair_reverse][:] = np.nan
    return d

def location(HS, HS_list, loc_path):
    loc_total = classification(HS_list, loc_path)
    temp = loc_total[str(HS)]
    combin = list(itertools.combinations(list(temp.keys()), 2))
    # 将脑区进行组合
    for brain_pairs in combin:
        if brain_pairs[0] in temp.keys() and brain_pairs[1] in temp.keys():
            elc_pair = list(itertools.product(temp[brain_pairs[0]], temp[brain_pairs[1]]))
            # 将脑区中的电极对进行组合，电极对与脑区在brain_pairs和area_pair的相同索引处
            elc_pair = np.array([[x, y] for x, y in elc_pair])
            yield brain_pairs, elc_pair



def prepare(dic,method):
    # 将字典按照任务合并，对音节求平均
    # 处理电极对的字典
    if method=='mean':
        flag = 'overt'
        dic_all = {}

        temp_1 = np.array([list(dic[list(dic.keys())[0]].values())])
        for k in list(dic.keys())[1:]:  # TASK_sy
            k_list = list(k.split('_'))
            if k_list[1] == flag:
                temp_1 = np.concatenate((temp_1, np.array([list(dic[k].values())])), axis=0)  # 合并成很多行的矩阵
                continue
            else:
                temp_1 = np.array([list(dic[k].values())])
                average = np.mean(temp_1, axis=0)  # 按照任务求平均值
                dic_all[flag] = average
                flag = k_list[1]
                continue
        average = np.mean(temp_1, axis=0)
        dic_all[flag] = average
        key_data = list(dic[list(dic.keys())[0]].keys())
        return dic_all, key_data
    elif method=='p':
        key_data = list(dic.keys())
        dic_all=np.array(list(dic.values()))
        return dic_all,key_data




def elec_sig(HS, mat_path):  # 生成显著电极索引
    elec_path = mat_path + "/" + str(HS) + "sig_elecs.npy"
    sig_elecs = np.load(elec_path, allow_pickle=True).item()
    for key in sig_elecs.keys():
        sig_elecs[key] = np.array(sig_elecs[key])
    return(sig_elecs)


def combine_by_task(d, task, data):
    # 将不同任务的数据添加进字典，并防止键值的覆盖
    # d: {task:{region_pairs}}
    # data: {region_pairs:}
    while True:
        if task in d:
            for ke, va in data.items():
                if ke in d[task]:
                    d[task][ke] = np.concatenate((va, d[task][ke]))
                else:
                    d[task][ke] = np.array(va)
                    continue
            break
        else:
            d[task] = {}
            continue
    return d


def calcu_median(dic):
    # 计算每个脑区对的中位数
    for task in dic.keys():
        for pair in dic[task].keys():
            dic[task][pair] = np.median(dic[task][pair])
    return dic


def lag_process_data(HS_list,loc_path, mat_path, clear_path, remove_dict):
    #初始化存储lag和r值的字典
    time={'overt': {}, 'covert': {}, 'cue': {}}
    alldata={'overt': {}, 'covert': {}, 'cue': {}}

    for HS in HS_list:
        #r和p值，不需要对音节求平均
        #读入文件并生成字典
        p_sound_avg_path = f"/HS{HS}_Block_lag_p_sound_avg.npy"
        r_sound_avg_path = f"/HS{HS}_Block_lag_r_sound_avg.npy"
        p_sound_avg = np.load(clear_path + p_sound_avg_path, allow_pickle=True).item()
        r_sound_avg = np.load(clear_path + r_sound_avg_path, allow_pickle=True).item()#键是三个任务

        elec_sig_dict = elec_sig(HS, mat_path)#生成显著电极的字典

        for task in r_sound_avg.keys():
            remove_list = remove_dict[task]
            sig = elec_sig_dict[task]
            x, y = np.meshgrid(sig, sig)
            sig_coordinates = np.column_stack((x.ravel(), y.ravel()))
            elec_pair_dict_pred_r, key_data_r = prepare(r_sound_avg[task], 'p')
            elec_pair_dict_pred_p, key_data_p = prepare(p_sound_avg[task], 'p')  # keyp是用不到的

            #下列elec_pair_dict_pred为电极对数n乘151个时间点的矩阵，原中心时间点center_tp为75
            center_tp = 75
            time_dis = 75

            elec_pairs_r = elec_pair_dict_pred_r[:, 75 - time_dis:75 + time_dis + 1]
            elec_pairs_p = elec_pair_dict_pred_p[:, 75 - time_dis:75 + time_dis + 1]

#电极对索引列表
            key_list = [pair.split('_') for pair in list(key_data_r)]
            last_key = key_list[-1][-1]
            key_list = np.array([[int(x), int(y)] for [x, y] in key_list])

            #新的r值
            data_r_r=np.zeros((int(last_key) + 1, int(last_key) + 1))
            data_pairs=np.zeros((int(last_key) + 1, int(last_key) + 1))

            #筛选掉p值>0.05的电极对
            #生成掩码
            mask_p=elec_pairs_p >= 0.05

            elec_pairs_r[mask_p] = np.nan
            r_r_value=np.amax(elec_pairs_r, axis=1)


            # 最大correlation的索引换算成时间lag
            elec_lag_r = (np.argmax(elec_pairs_r, axis=1) - center_tp) / 100

        #赋值
            data_pairs[key_list[:, 0], key_list[:, 1]]=elec_lag_r
            data_r_r[key_list[:, 0], key_list[:, 1]]=r_r_value

            # 存储索引和r值
            # 生成对称矩阵方便提取数值
            data_pairs=data_pairs-np.triu(data_pairs, 1).T
            data_r_r=data_r_r+np.triu(data_r_r, 1).T

            #生成脑区对
            location0 = location(HS, HS_list, loc_path)


            #滤除不显著电极
            mask = np.zeros_like(data_r_r, dtype=bool)
            mask[sig_coordinates[:, 0], sig_coordinates[:, 1]] = True
              # 用掩码去除非显著电极
            data_pairs[~mask] = np.nan
            data_r_r[~mask] = np.nan

            mask_r = data_r_r < 0.2
            data_r_r[mask_r] = np.nan
            data_pairs[mask_r] = np.nan

            for counter in range(10000):
                try:
                    tags, elc_pair = next(location0)
                    #按照脑区筛选
                    datalist = data_pairs[elc_pair[:, 0], elc_pair[:, 1]]
                    rlist = data_r_r[elc_pair[:, 0], elc_pair[:, 1]]
                    #过滤
                    datalist_filtered = datalist[~np.isnan(datalist)]
                    rlist_filtered = rlist[~np.isnan(rlist)]

                    if np.size(datalist_filtered) != 0:
                        #删除脑区对
                        if tags[0] not in remove_list and tags[1] not in remove_list:
                            if f'{tags[0]}' + '_' + f'{tags[1]}' in alldata[task]:
                                #按照任务将相同的脑区对合并成一个矩阵
                                alldata[task][f'{tags[0]}' + '_' + f'{tags[1]}'] = np.concatenate(
                                    (alldata[task][f'{tags[0]}' + '_' + f'{tags[1]}'], rlist_filtered), axis=0)#r值
                                time[task][f'{tags[0]}' + '_' + f'{tags[1]}'] = np.concatenate(
                                    (time[task][f'{tags[0]}' + '_' + f'{tags[1]}'], datalist_filtered), axis=0)#lag值
                            else:
                                alldata[task][f'{tags[0]}' + '_' + f'{tags[1]}'] = rlist_filtered
                                time[task][f'{tags[0]}' + '_' + f'{tags[1]}'] = datalist_filtered
                        else:
                            continue
                except StopIteration:
                    break

    for task in r_sound_avg.keys():#tasks
        # 进行去重
        time[task] = prepare_dict(time[task], 'lag')#相反取负
        alldata[task] = prepare_dict(alldata[task], 'r')#相反不变

    return alldata,time


def boxplot_data(time,width):
    for task in time.keys():
        if time[task]:
            copy_dict =copy.deepcopy(time[task])
            for pairs in copy_dict.keys():
                if np.median(copy_dict[pairs]) > 0:
                    time[task][pairs] = -time[task][pairs]
                    left, right = pairs.split("_")
                    new_pairs = right + "_" + left
                    time[task][new_pairs] = time[task].pop(pairs)

            time[task] = dict(sorted(time[task].items(), key=lambda x: np.median(x[1]), reverse=True))

            ticks = [m.split('_') for m in time[task].keys()]
            tk2 = [n[1] for n in ticks]
            tk1 = [n[0] for n in ticks]

            fig, ax = plt.subplots(figsize=(17, 17))
            ax2 = ax.twinx()

            sns.boxplot(data=list(time[task].values()), showfliers=False, ax=ax, color='skyblue',
                        orient='horizontal', width=width)
            ax.set_yticklabels(tk1, fontsize=12)
            ax2.set_yticks(range(len(tk2)))
            ax2.set_yticklabels(tk2, fontsize=12)
            ax2.set_ylim(ax.get_ylim())
            ax.set_title(f'{task}', fontsize=24)
            ax.set_xlabel('Time lag(S)')
            ax.yaxis.set_label_coords(0.001, 1.05)
            ax2.yaxis.set_label_coords(1.02, 1.05)
            ax.axvline(x=0, color='black')
            plt.show()

#########################new_lag_data_process
def initialize_data(clean_path,HS,task):
    r=np.load(clean_path+f"lags/{task}/HS{HS}_combined/r_fold_combined.npy",allow_pickle=True).item()
    r2=np.load(clean_path+f"lags/{task}/HS{HS}_combined/r2_fold_combined.npy",allow_pickle=True).item()
    wts=np.load(clean_path+f"lags/{task}/HS{HS}_combined/wts_fold_combined.npy",allow_pickle=True).item()
    return r,r2,wts

def fill_up_matrix(dic):
    matrix=np.zeros((256,256))

    key_list = [pair.split('_') for pair in list(dic.keys())]
    indices = np.array([[int(x), int(y)] for [x, y] in key_list])
    mask=np.ones_like(matrix,dtype=bool)
    mask[indices[:,0],indices[:,1]] = False

    matrix[indices[:,0],indices[:,1]]=np.array(list(dic.values()))
    matrix[mask]=np.nan

    return matrix

def filter(r2_matrix,wts_matrix,r_matrix):
    mask=r2_matrix < 0.2
    wts_matrix[~mask]=np.nan
    r2_matrix[~mask]=np.nan
    r_matrix[~mask]=np.nan

    return wts_matrix,r2_matrix,r_matrix

def fill_up_dic(dic_all,matrix,key):
    if key in dic_all:
        dic_all[key]=np.concatenate((dic_all[key],matrix),axis=0)
    else:
        dic_all[key]=matrix
    return dic_all

def brain_area_selection(HS,HS_list,loc_path,mat,task,alldata,type):
    location0 = location(HS, HS_list, loc_path)
    for counter in range(10000):
        try:
            brain_pair, elc_pair = next(location0)
            # 按照脑区筛选
            selected_mat = mat[elc_pair[:, 0], elc_pair[:, 1]]
            # 过滤
            selection_filtered = selected_mat[~np.isnan(selected_mat)]

            if np.size(selection_filtered) != 0:
                # 删除脑区对
                    if brain_pair[0]+ '_' + brain_pair[1] in alldata[task]:
                        # 按照任务将相同的脑区对合并成一个矩阵
                        alldata[task][brain_pair[0]+ '_' + brain_pair[1]] = np.concatenate(
                            (alldata[task][brain_pair[0]+ '_' + brain_pair[1]], selection_filtered), axis=0)
                    else:
                        alldata[task][brain_pair[0]+ '_' + brain_pair[1]] = selection_filtered
            else:
                continue
        except StopIteration:
            break

    alldata[task]=prepare_dict(alldata[task],type)
    return alldata

def data_process(clean_path,HS_list,task_list,loc_path):

    r_value= {}
    r2_value={}
    wts_max_index={}

    r_value_brain_pair={'overt':{},'covert':{},'cue':{}}
    r2_value_brain_pair={'overt':{},'covert':{},'cue':{}}
    wts_max_index_brain_pair={'overt':{},'covert':{},'cue':{}}


    for task in task_list:
        for HS in HS_list:

            r,r2,wts=initialize_data(clean_path,HS,task)

            center_tp = 75

            r_value_temp={key:np.mean(value,axis=0)[0]for key,value in r.items()}
            r2_value_temp={key:np.mean(value,axis=0)[0]for key,value in r2.items()}
            wts_max_index_temp={key:(np.argmax(np.mean(value,axis=0))-center_tp)/100 for key,value in wts.items()}

            wts_data=fill_up_matrix(wts_max_index_temp)
            r_data=fill_up_matrix(r_value_temp)
            r2_data=fill_up_matrix(r2_value_temp)
            wts_data,r2_data,r_data=filter(r2_data,wts_data,r_data)

            r_value_brain_pair=brain_area_selection(HS,HS_list,loc_path,r_data,task,r_value_brain_pair,'r')
            r2_value_brain_pair=brain_area_selection(HS,HS_list,loc_path,r2_data,task,r2_value_brain_pair,'r')
            wts_max_index_brain_pair=brain_area_selection(HS,HS_list,loc_path,wts_data,task,wts_max_index_brain_pair,'lag')


            r_value=fill_up_dic(r_value,r_data,task)
            r2_value=fill_up_dic(r2_value,r2_data,task)
            wts_max_index=fill_up_dic(wts_max_index,wts_data,task)

    return wts_max_index,r_value,r2_value,r_value_brain_pair,r2_value_brain_pair, wts_max_index_brain_pair




if __name__ == '__main__':
    HS_list_1 = [44, 45, 47, 48, 50, 54, 71, 73, 76, 78]
    width = 0.2
    loc_path_1 = "E:/vs/python/data_for_code/HSblockdata/"
    mat_path = "E:/vs/python/data_for_code/HSblockdata/elecs/elec_sig"
    remove_dict = {'overt': [], 'covert': [], 'cue': []}
    clear_path = "E:/vs/python/lags"
    alldata,time=lag_process_data(HS_list_1,loc_path_1, mat_path, clear_path, remove_dict)
    boxplot_data(time,width)

    clean_path = "E:/vs/python/"
    loc_path = "E:/vs/python/data_for_code/HSblockdata/"
    HS_list=[45]
    task_list=['overt']
    wts_max_index,r_value,r2_value,r_value_brain_pair,r2_value_brain_pair, wts_max_index_brain_pair=data_process(clean_path,HS_list,task_list,loc_path)
    boxplot_data(wts_max_index_brain_pair,width)

