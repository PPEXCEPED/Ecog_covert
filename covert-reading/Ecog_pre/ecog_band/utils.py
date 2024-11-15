import numpy as np
import os
import  wave
import matplotlib.pyplot as plt
import json
from random import shuffle
import math
import mne
import torch   
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd
from sklearn.model_selection import learning_curve
import seaborn as sns
import ast
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import librosa


def get_HS():
    all_HS = ['62', '65', '68', '69', '75', '79', '82', '84', '85', '86']
    return all_HS

def get_all_band():
    all_band = ['else1','delta','theta','alpha','beta', 'gamma','high gamma','else2']
    return all_band

def get_all_elec():
    all_elec = np.arange(1, 126).tolist()
    return all_elec

def calculating_time(time,delay,freq):
    fs=24414
    time=time.to(torch.float)
    delay/=fs
    time/=1000
    time-=delay
    time=time*(freq/2)
    return time.to(torch.int)

def map_dist(dist):
    code_dist=[]
    code_list=['功课','数页','宫客','shù yè','力果','对十','树叶','gōng kè','作业','绿草']
    for word in dist:
        code_dist.append(code_list.index(word))
    return code_dist

def np_save(nparr,path):
    if os.path.exists(path):
        os.remove(path)
    with open(path,'wb')as file:
        np.save(file,nparr)

def calculating_time_domain(time,delay,freq):
    fs=24414
    time=time.to(torch.float)
    delay/=fs
    time/=1000
    time-=delay
    time=time*freq
    return time.to(torch.int)

def stftAndsplit(HS,PATH,freq_list,path_save,idx_elec,idx_block=0,forward_cue=0,backward_cue=0,forward_read=0,backward_read=0):#idx_elec is a list or a number:if list, traverse the list; else, transform it to a list
    path_time=os.path.join(PATH,'processed_data',f'HS{HS}')
    path_points=os.path.join(PATH,'points')
    cue=np.load(os.path.join(path_points,f'HS{HS}_oneset_cue_point.npy'),allow_pickle=True).item()#The files saved are all dics,so remeber to add item
    read=np.load(os.path.join(path_points,f'HS{HS}_onset_read_time.npy'),allow_pickle=True).item()#key is the block and value is the cue or read point (list) or the delay time(num)
    delays=np.load(os.path.join(path_points,f'HS{HS}_delay_list.npy'),allow_pickle=True).item()
    words=np.load(os.path.join(path_points,f'HS{HS}_words.npy'),allow_pickle=True).item()

    if isinstance(idx_elec, int):
        idx_elec = [idx_elec]

    for freq in freq_list:
        temp=os.path.join(path_time,str(freq))#ecog data in time domain
        path = os.listdir(temp)
        for z in range(idx_block,len(path)):
            
            ecog_block=np.load(os.path.join(temp,path[z]))
            oneset_cue_point=cue[str(z)]
            onset_read_time=read[str(z)]
            delay=delays[str(z)]
            word=words[str(z)]

            task_label=map_dist(word)#list
            oneset_cue=calculating_time(oneset_cue_point,delay.item(),freq)
            read_time=calculating_time(onset_read_time,delay.item(),freq)
            
            time=np.zeros(ecog_block.shape[1])
            task_time=np.zeros(ecog_block.shape[1])

            for cnt_for_words_cue,index in enumerate(oneset_cue):
                time[index:index+int(freq/2)]=1#time line with label,a 1 dim vector
                task_time[index:index+int(freq/2)]=task_label[cnt_for_words_cue]#10 word classification

            for cnt_for_words_read,id in enumerate(read_time):
                time[id:id+int(id+freq*0.75)]=2
                task_time[id:id+int(id+freq*0.75)]=task_label[cnt_for_words_read]

            for elec in idx_elec:
                print(f'This is freq:{freq},block:{z},elec:{elec}')

                data_block_cue=[]
                data_block_read=[]
                ecog_block_seg=mne.time_frequency.stft(ecog_block[elec],2*freq,2)
                print(ecog_block_seg.shape)
                print(ecog_block_seg[:,:,oneset_cue[0]:].shape)
                ecog_block_seg=(ecog_block_seg-np.mean(ecog_block_seg[:,:,oneset_cue[0]:],axis=2,keepdims=True))/np.std(ecog_block_seg[:,:,oneset_cue[0]:],axis=2,keepdims=True)
                for index in oneset_cue:
                    stft_block_cue=ecog_block_seg[0,:,index-forward_cue:index+int(freq/2)+backward_cue]#the first dimension will be vanished
                    print(stft_block_cue.shape)
                    if stft_block_cue.shape[1]==int(freq/2)+forward_cue+backward_cue:
                        data_block_cue.append(stft_block_cue)#spectrum block with 256 elecs
                    
                for id in read_time:
                    stft_block_read=ecog_block_seg[0,:,id-forward_read:int(id+freq*0.75)+backward_read]
                    print(stft_block_read.shape)
                    if stft_block_read.shape[1]==int(freq*0.75)+forward_read+backward_read:
                        data_block_read.append(stft_block_read)
                    

                data_block_cue=np.stack(data_block_cue)
                data_block_read=np.stack(data_block_read)
                
                os.makedirs(os.path.join(path_save,f'dataset/HS{HS}',str(freq),str(elec)),exist_ok=True)
                path_elec=os.path.join(path_save,f'dataset/HS{HS}',str(freq),str(elec))
                print('DONE')
                print(data_block_cue.shape,data_block_read.shape)
                np_save(data_block_cue,os.path.join(path_elec,f'elec{elec}_{z}_data_block_cue.npy'))
                np_save(data_block_read,os.path.join(path_elec,f'elec{elec}_{z}_data_block_read.npy'))
                
            task_label=np.array(task_label)
            np_save(task_label,os.path.join(path_save,f'dataset/HS{HS}',str(freq),f'{z}_task_label.npy'))
            np_save(time,os.path.join(path_save,f'dataset/HS{HS}',str(freq),f'{z}_time.npy'))
            np_save(task_time,os.path.join(path_save,f'dataset/HS{HS}',str(freq),f'{z}_task_time.npy'))
        
def pltbox_band_five_cross_validation(band_acc_list, permutation=False):
    freq_bands = band_acc_list.keys()
    band_accs = list(band_acc_list.values())

    cleaned_band_accs = []
    for acc_list in band_accs:
        if acc_list is not None:
            cleaned_band_accs.append(acc_list)
        else:
            # 处理None值，例如用空列表替换
            cleaned_band_accs.append([])

    plt.boxplot(cleaned_band_accs)
    plt.xticks(ticks=range(1, len(freq_bands) + 1), labels=freq_bands)
    if permutation == True:
        plt.ylabel('Accuracy(permutation)')
    else:
        plt.ylabel('Accuracy')
    plt.title('Accuracy Distribution Across Frequency Bands')
    plt.grid(True)
    plt.show()

def save_band_classify_acc(band_acc_list, save_path, permutation=False):
    # save_path = '/root/pp/covert-reading/Ecog_pretrain/fold_results'
    os.makedirs(save_path, exist_ok=True)
    if permutation==True:
        file_path = os.path.join(save_path, 'permut_results.txt')
    else:
        file_path = os.path.join(save_path, 'general_results.txt')
    with open(file_path, 'w') as file:
        freq_bands = list(band_acc_list.keys())
        band_accs = list(band_acc_list.values())
        for i in range(len(freq_bands)):
            band_acc = band_accs[i]
            # 将每个 numpy array 转换为浮点数并转为字符串
            band_acc_str = ' '.join(map(lambda x: str(float(x)), band_acc))
            line = f"{freq_bands[i]} {band_acc_str}"
            file.write(line + '\n')

def plt_confusion_matric(y_pred, y_true, HS, elec, freq, band='All_band'):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    # 绘制混淆矩阵
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix({band}-HS{HS}-elec{elec}-sr{freq})')
    plt.colorbar()
    tick_marks = np.arange(len(set(y_true)))
    plt.xticks(tick_marks, set(y_true), rotation=45)
    plt.yticks(tick_marks, set(y_true))

    # 在每个单元格中添加数值
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plt_confusion_matrix_sum(y_true_list, y_pred_list, band):# 计算五折的和的混淆矩阵
    # 计算混淆矩阵
    # all_y_true = [label for sublist in y_true_list for label in sublist]
    # all_y_pred = [label for sublist in y_pred_list for label in sublist]
    cm = confusion_matrix(y_true_list, y_pred_list)
    # cm = confusion_matrix(y_true_list[0], y_pred_list[0])
    # for i in range(len(y_pred_list)):
    #     if i==0:
    #         continue 
    #     cm += confusion_matrix(y_true_list[i], y_pred_list[i])

    # 绘制混淆矩阵
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix({band})')
    plt.colorbar()
    tick_marks = np.arange(len(set(y_true_list)))
    plt.xticks(tick_marks, set(y_true_list), rotation=45)
    plt.yticks(tick_marks, set(y_true_list))

    # 在每个单元格中添加数值
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def cal_confusion_matrix_sum(y_true_list, y_pred_list):
    all_y_true = [label for sublist in y_true_list for label in sublist]
    all_y_pred = [label for sublist in y_pred_list for label in sublist]
    cm = confusion_matrix(all_y_true, all_y_pred)
    return cm
    
def cal_acc_sum(band_acc_list):
    freq_bands = list(band_acc_list.keys())
    band_accs = list(band_acc_list.values())
    band_acc = [float(np.mean(sublist)) for sublist in band_accs]
    return band_acc

def cal_acc_band1_from_y(y_path, band):
    y_true = np.load(os.path.join(y_path, f'{band}_y_true.npy'))
    y_pred = np.load(os.path.join(y_path, f'{band}_y_pred.npy'))
    return accuracy_score(y_true, y_pred)

def cal_cm_band1_from_y(y_path, band):
    y_true = np.load(os.path.join(y_path, f'{band}_y_true.npy'))
    y_pred = np.load(os.path.join(y_path, f'{band}_y_pred.npy'))
    return confusion_matrix(y_true, y_pred)

def cal_report_from_y(y_path, band):
    y_true = np.load(os.path.join(y_path, f'{band}_y_true.npy'))
    y_pred = np.load(os.path.join(y_path, f'{band}_y_pred.npy'))
    report = classification_report(y_true, y_pred, target_names=['看', '读'])
    return report

def pltbar_contribution(contribution, band_list, HS, elec, freq, fig_save_path=None, removed=False):
    max_contribution_band = np.argmax(contribution)
    colors = ['red' if x < 0 else 'blue' for x in contribution]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(band_list, contribution)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # X coordinate (center of the bar)
            height,  # Y coordinate (top of the bar)
            f'{height*100:.1f}%',  # Text to display
            ha='center',  # Horizontal alignment
            va='bottom',  # Vertical alignment
            fontsize = 8,
        )

    max_contribution_band = np.argmax(contribution)
    plt.plot(
        band_list[max_contribution_band], 
        contribution[max_contribution_band], 
        '*',  # Marker style
        color='red', 
        markersize=10, 
        # label=f'Max Accuracy: {accuracy[max_accuracy_band]*100:.2f}%'
    )
    plt.legend()

    plt.xlabel(f'Frequency Band {"(removed)" if removed else ""}')
    plt.ylabel('Contribution to Accuracy')
    # plt.title('Contribution of Each Frequency Band')
    plt.title(f'HS{HS}-elec{elec}-sr{freq}')
    plt.margins(y=0.02) 
    plt.xticks(rotation=45)
    plt.tight_layout()

    if fig_save_path != None:
        plt.savefig(os.path.join(fig_save_path, f'contri-HS{HS}-elec{elec}-sr{freq}.png'), dpi=300)
    
    # 显示图像
    plt.show()
    plt.clf()
    plt.close()

def pltbar_accuracy(accuracy, band_list, HS, freq, elec, fig_save_path=None, removed=False):
    max_accuracy_band = np.argmax(accuracy)
    colors = ['red' if x < 0 else 'blue' for x in accuracy]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(band_list, accuracy)
    # plt.annotate(
    #     f'Max Accuracy: {band_list[max_accuracy_band]}\n{accuracy[max_accuracy_band]:.2f}',
    #     xy=(band_list[max_accuracy_band], accuracy[max_accuracy_band]),
    #     xytext=(band_list[max_accuracy_band], accuracy[max_accuracy_band] + 0.1),
    #     arrowprops=dict(facecolor='black', arrowstyle='->'),
    #     fontsize=10,
    #     ha='center',
    # )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # X coordinate (center of the bar)
            height,  # Y coordinate (top of the bar)
            f'{height*100:.1f}%',  # Text to display
            ha='center',  # Horizontal alignment
            va='bottom',  # Vertical alignment
            fontsize=8
        )
    
    max_accuracy_band = np.argmax(accuracy)
    plt.plot(
        band_list[max_accuracy_band], 
        accuracy[max_accuracy_band], 
        '*',  # Marker style
        color='red', 
        markersize=10, 
        # label=f'Max Accuracy: {accuracy[max_accuracy_band]*100:.2f}%'
    )
    plt.legend()

    plt.xlabel(f'Frequency Band {"(removed)" if removed else ""}')
    plt.ylabel('Accuracy')
    plt.title(f'HS{HS}-elec{elec}-sr{freq}')
    plt.margins(y=0.05) 
    plt.xticks(rotation=45)
    plt.tight_layout()

    
    if fig_save_path != None:
        if removed == False:
            plt.savefig(os.path.join(fig_save_path, f'acc-HS{HS}-elec{elec}-sr{freq}.png'), dpi=300)
        else:
            plt.savefig(os.path.join(fig_save_path, f'accrem-HS{HS}-elec{elec}-sr{freq}.png'), dpi=300)
            
    # 显示图像
    plt.show()
    plt.clf()
    plt.close()

def plt_band_acc(band_acc_list):
    freq_bands = list(band_acc_list.keys())
    band_accs = list(band_acc_list.values())
    band_accs = [float(np.mean(sublist)) for sublist in band_accs]

    plt.figure(figsize=(12, 6))
    plt.bar(freq_bands, band_accs)
    plt.xlabel('Frequency Band')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Each Frequency Band')

    # # 添加每个条形的标签
    # for i, v in enumerate(contribution_df['contribution']):
    #     plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
    plt.axhline(0.5, color='gray', linewidth=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 显示图像
    plt.show()
def plt_learning_curve(model, X, y, cv=5):
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )

    # 计算平均值和标准差
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    # 绘制学习曲线
    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.grid()

    # 绘制训练分数
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.fill_between(train_sizes, 
                     train_scores_mean - train_scores_std, 
                     train_scores_mean + train_scores_std, 
                     alpha=0.1, color='r')

    # 绘制验证分数
    plt.plot(train_sizes, valid_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(train_sizes, 
                     valid_scores_mean - valid_scores_std, 
                     valid_scores_mean + valid_scores_std, 
                     alpha=0.1, color='g')

    plt.legend(loc='best')
    plt.show()
# def plt_learning_curve(clf, X, y):
#     train_sizes, train_scores, test_scores = learning_curve(
#         clf, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    
#     plt.figure()
#     plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
#     plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
#     plt.xlabel('Number of Training Examples')
#     plt.ylabel('Score')
#     plt.title('Learning Curve')
#     plt.legend(loc='best')
#     plt.grid()
#     plt.show()

def plt_allband_confusion_matrices(confusion_matrices, HS, elec, freq, bands, fig_save_path=None):
    num_bands = len(bands)
    fig, axes = plt.subplots(nrows=(num_bands + 2) // 3, ncols=3, figsize=(15, (num_bands // 3 + 1) * 5))
    axes = axes.flatten()

    for i, band in enumerate(bands):
        cm = confusion_matrices[band]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Predicted Negative', 'Predicted Positive'], 
                    yticklabels=['Actual Negative', 'Actual Positive'])
        axes[i].set_title(band)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout
    plt.title(f'HS{HS}-elec{elec}-sr{freq}')
    plt.tight_layout()
    
    if fig_save_path!=None:
        plt.savefig(os.path.join(fig_save_path, f'cm-HS{HS}-elec{elec}-sr{freq}.png'), dpi=300)
    
    plt.show()
    plt.clf()
    plt.close()

def normalise_stft_data(stft_data):
    # 将数据展平为二维数组 (n_task, n_freq * n_timePoint)
    data_2d = stft_data.reshape(-1, stft_data.shape[1] * stft_data.shape[2])
    # 选择 MinMaxScaler 或 StandardScaler
    scaler = StandardScaler()  # 或者使用 StandardScaler()
    data_normalized = scaler.fit_transform(data_2d)

    # 将数据重新恢复到原始形状
    elec_cue_normalized = data_normalized.reshape(stft_data.shape)

# def z_score_ecog(ecog_mat):
#     # Calculate the mean and standard deviation along the last dimension
#     mean = np.nanmean(ecog_mat, axis=-1, keepdims=True)
#     std = np.nanstd(ecog_mat, axis=-1, keepdims=True)
    
#     # Z-score the data
#     z_scored_data = (ecog_mat - mean) / std
    
#     return z_scored_data

# def calcu_spec(raw_mat,n_mels = 128,fmin = 0.3,fmax = 1500):
#     n_fft = 2048  # FFT window size
#     hop_length = int(3052 / 100)  # Hop length to achieve 100 Hz sampling rate
#     spec = np.zeros((n_mels,raw_mat.shape[0],raw_mat.shape[1]))
    
#     for pidx,clip in enumerate(raw_mat):
#         # Compute the spectrogram without resampling
#         S = librosa.feature.melspectrogram(y=clip, sr=3052, n_fft=n_fft, hop_length=hop_length,
#                                            n_mels=n_mels, fmin=fmin, fmax=fmax)
#         S_dB = librosa.power_to_db(S, ref=np.max)

#         # Display the spectrogram
#         #librosa.display.specshow(S_dB, sr=3052, hop_length=hop_length, x_axis='time', y_axis='mel')
#         #plt.colorbar(format='%+2.0f dB')
#         #plt.show()
        
#         # S_dB = downsample(S_dB, tsr=len(clip), osr=S_dB.shape[1])
#         S_dB = z_score_ecog(S_dB)
#         spec[:,pidx,:] = S_dB
#     #print(spec.shape)
#     return(spec)

def cal_all_elec_acc(rootPath, HS, freq):
    acc_dic = {}
    band_list = get_all_band()
    idx_elec = list(range(0, 256))
    
    for elecidx in idx_elec:
        band_acc = {}
        for band in band_list:
            # y_true_path = os.path.join(rootPath, f'HS{HS}/{freq}/{elecidx}/{band}_y_true.npy')
            # y_pred_path = os.path.join(rootPath, f'HS{HS}/{freq}/{elecidx}/{band}_y_pred.npy')
            acc = cal_acc_band1_from_y(os.path.join(rootPath, str(elecidx)), band)
            band_acc[band] = acc
        acc_dic[elecidx] = band_acc
        # print(f'elec{elecidx} acc_list:{band_acc}')
    return acc_dic

def cal_all_elec_cm(rootPath, HS, freq):
    cm_dic = {}
    band_list = get_all_band()
    idx_elec = list(range(0, 256))
    for elecidx in idx_elec:
        band_cm = {}
        for band in band_list:
            # y_true_path = os.path.join(rootPath, f'HS{HS}/{freq}/{elecidx}/{band}_y_true.npy')
            # y_pred_path = os.path.join(rootPath, f'HS{HS}/{freq}/{elecidx}/{band}_y_pred.npy')
            acc = cal_cm_band1_from_y(os.path.join(rootPath, str(elecidx)), band)
            band_cm[band] = acc
        cm_dic[elecidx] = band_cm
    return cm_dic
    